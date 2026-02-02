# train_dynamics.py
import os
import time
import math
import random
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import einops

import wandb

from miniconf import MiniConf
from wm_dataset import WMDataset, collate_batch, split_by_trajectory
from model import (
    Tokenizer,
    Dynamics,
    temporal_patchify, temporal_unpatchify,
    pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)


def is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def seed_everything(seed: int):
    s = int(seed) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    seed_everything(info.seed)


def init_distributed() -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return ddp, rank, world_size, local_rank


@torch.no_grad()
def load_frozen_tokenizer(ckpt_path: str, device: torch.device) -> Tokenizer:
    """Load a frozen tokenizer from checkpoint."""
    tok, _ = Tokenizer.from_checkpoint(Path(ckpt_path))
    tok = tok.to(device)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


# ---- Evaluation helpers ----

def _is_pow2(n: int) -> bool:
    return (n > 0) and ((n & (n - 1)) == 0)


def make_tau_schedule(*, k_max: int, schedule: str, d: Optional[float] = None) -> Dict[str, Any]:
    """
    Returns a schedule dict for sampling:
      K = number of integration steps
      e = log2(K)
      tau_idx[i] = discrete signal index at step i
      tau[i] = i/K
      dt = 1/K
    """
    assert _is_pow2(k_max), "k_max must be power of two"
    if schedule == "finest":
        K = k_max
    elif schedule == "shortcut":
        assert d is not None, "shortcut schedule requires d"
        inv = int(round(1.0 / float(d)))
        assert _is_pow2(inv), "d must be 1/(power of two)"
        assert inv <= k_max, "d must be >= 1/k_max"
        K = inv
    else:
        raise ValueError(f"unknown schedule: {schedule}")

    e = int(round(math.log2(K)))
    scale = k_max // K
    tau = [i / K for i in range(K)] + [1.0]
    tau_idx = [i * scale for i in range(K)] + [k_max]
    return dict(K=K, e=e, scale=scale, tau=tau, tau_idx=tau_idx, dt=1.0 / K, schedule=schedule, d=1.0 / K)


@torch.no_grad()
def sample_one_timestep(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,
    act_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample a single future timestep given past context."""
    device = past_packed.device
    dtype = past_packed.dtype
    B, t = past_packed.shape[:2]
    n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]

    K = int(sched["K"])
    e = int(sched["e"])
    tau = sched["tau"]
    tau_idx = sched["tau_idx"]
    dt = float(sched["dt"])

    # Start from noise
    z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)

    emax = int(round(math.log2(k_max)))
    step_idxs_full = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
    step_idxs_full[:, -1] = e

    signal_idxs_full = torch.full((B, t + 1), k_max - 1, device=device, dtype=torch.long)

    if act_mask is not None and act_mask.dim() == 1:
        act_mask = act_mask.view(1, 1, -1)

    for i in range(K):
        tau_i = float(tau[i])
        sig_i = int(tau_idx[i])

        signal_idxs_full[:, -1] = sig_i
        packed_seq = torch.cat([past_packed, z], dim=1)

        actions_in = None if actions is None else actions[:, :t + 1]
        actmask_in = None if act_mask is None else act_mask[:, :t + 1]

        x1_hat_full, _ = dyn(
            actions_in, step_idxs_full, signal_idxs_full, packed_seq,
            act_mask=actmask_in, agent_tokens=None,
        )
        x1_hat = x1_hat_full[:, -1:, :, :]

        denom = max(1e-4, 1.0 - tau_i)
        b = (x1_hat.float() - z.float()) / denom
        z = (z.float() + b * dt).to(dtype)

    return z[:, 0]


@torch.no_grad()
def sample_autoregressive(
    dyn: Dynamics,
    *,
    z_gt_packed: torch.Tensor,
    ctx_length: int,
    horizon: int,
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,
    act_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Autoregressively sample future frames."""
    B, T = z_gt_packed.shape[:2]
    L = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, L - 1)
    horizon = min(horizon, L - ctx_length)

    outs = [z_gt_packed[:, t] for t in range(ctx_length)]

    for t in range(ctx_length, ctx_length + horizon):
        past = torch.stack(outs, dim=1)
        z_next = sample_one_timestep(
            dyn, past_packed=past, k_max=k_max, sched=sched,
            actions=actions, act_mask=act_mask,
        )
        outs.append(z_next)

    return torch.stack(outs, dim=1)


@torch.no_grad()
def decode_to_frames(
    tokenizer: Tokenizer,
    z_packed: torch.Tensor,
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
) -> torch.Tensor:
    """Decode packed latents to frames."""
    z_btLd = unpack_spatial_to_bottleneck(z_packed, k=packing_factor)
    patches_btnd = tokenizer.decode(z_btLd)
    frames = temporal_unpatchify(patches_btnd, H, W, C, patch)
    return frames.clamp(0, 1)


def compute_horizon_metrics(
    gt: torch.Tensor, 
    pred: torch.Tensor, 
    ctx_length: int,
    horizons: List[int] = [1, 4, 8, 16]
) -> Dict[str, float]:
    """
    Compute MSE at specific prediction horizons.
    
    Args:
        gt: Ground truth frames (B, T, C, H, W)
        pred: Predicted frames (B, T, C, H, W) 
        ctx_length: Number of context frames
        horizons: List of horizons to evaluate (relative to ctx_length)
        
    Returns:
        Dictionary of metrics for each horizon
    """
    results = {}
    T_avail = pred.shape[1] - ctx_length
    
    for h in horizons:
        if h <= T_avail:
            # Get frames at horizon h (relative to end of context)
            gt_h = gt[:, ctx_length:ctx_length + h]
            pred_h = pred[:, ctx_length:ctx_length + h]
            mse = (pred_h.float() - gt_h.float()).pow(2).mean()
            psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
            results[f"eval/mse_t{h}"] = float(mse.item())
            results[f"eval/psnr_t{h}"] = float(psnr.item())
    
    return results


@torch.no_grad()
def run_eval(
    *,
    tokenizer: Tokenizer,
    dyn: Dynamics,
    frames: torch.Tensor,
    actions: Optional[torch.Tensor],
    act_mask: Optional[torch.Tensor],
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
    k_max: int,
    ctx_length: int,
    horizon: int,
    sched: Dict[str, Any],
    max_items: int,
    step: int,
):
    """Run evaluation and log to wandb."""
    dyn_was_training = dyn.training
    dyn.eval()

    B, T = frames.shape[:2]
    T_eval = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, T_eval - 1)
    horizon = min(horizon, T_eval - ctx_length)

    frames_eval = frames[:, :T_eval]

    # Encode to latents
    patches = temporal_patchify(frames_eval, patch)
    z_btLd = tokenizer.encode(patches)
    n_spatial = z_btLd.shape[2] // packing_factor
    z_gt_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)

    actions_eval = None if actions is None else actions[:, :T_eval]
    act_mask_eval = None if act_mask is None else (act_mask[:, :T_eval] if act_mask.dim() == 3 else act_mask)

    # Sample predictions
    z_pred_packed = sample_autoregressive(
        dyn, z_gt_packed=z_gt_packed, ctx_length=ctx_length, horizon=horizon,
        k_max=k_max, sched=sched, actions=actions_eval, act_mask=act_mask_eval,
    )

    pred_frames = decode_to_frames(tokenizer, z_pred_packed, H, W, C, patch, packing_factor)

    # Floor baseline: repeat last context frame
    floor = frames_eval.clone()
    if horizon > 0:
        floor[:, ctx_length:ctx_length + horizon] = frames_eval[:, ctx_length - 1:ctx_length].expand(-1, horizon, -1, -1, -1)

    # Compute metrics on horizon only
    gt_h = frames_eval[:, ctx_length:ctx_length + horizon]
    pred_h = pred_frames[:, ctx_length:ctx_length + horizon]
    floor_h = floor[:, ctx_length:ctx_length + horizon]

    mse_pred = (pred_h.float() - gt_h.float()).pow(2).mean()
    mse_floor = (floor_h.float() - gt_h.float()).pow(2).mean()

    psnr_pred = 10.0 * torch.log10(1.0 / mse_pred.clamp_min(1e-12))
    psnr_floor = 10.0 * torch.log10(1.0 / mse_floor.clamp_min(1e-12))

    mse_ratio = mse_pred / mse_floor.clamp_min(1e-12)
    psnr_gain = psnr_pred - psnr_floor

    metrics = {
        "eval/mse_pred": float(mse_pred.item()),
        "eval/mse_floor": float(mse_floor.item()),
        "eval/mse_ratio": float(mse_ratio.item()),
        "eval/psnr_pred": float(psnr_pred.item()),
        "eval/psnr_floor": float(psnr_floor.item()),
        "eval/psnr_gain_db": float(psnr_gain.item()),
    }
    
    # Add multi-horizon metrics
    horizon_metrics = compute_horizon_metrics(frames_eval, pred_frames, ctx_length, horizons=[1, 4, 8, 16])
    metrics.update(horizon_metrics)

    if horizon > 0:
        wandb.log(metrics, step=step)

    # Log visualization
    log_viz(gt=frames_eval, pred=pred_frames, ctx_length=ctx_length, step=step, max_items=max_items)

    if dyn_was_training:
        dyn.train()


@torch.no_grad()
def run_validation_dynamics(
    *,
    dyn: Dynamics,
    tokenizer: Tokenizer,
    val_loader: DataLoader,
    device: torch.device,
    packing_factor: int,
    use_actions: bool,
    step: int,
    max_batches: int = 25,
) -> Dict[str, float]:
    """
    Run validation on held-out data for dynamics model.
    
    Returns:
        Dictionary of validation metrics
    """
    dyn_was_training = dyn.training
    dyn.eval()
    
    total_flow_mse = 0.0
    total_samples = 0
    
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        
        # Prepare data (same as training)
        if use_actions and "act" in batch:
            obs_u8 = batch["obs"].to(device, non_blocking=True)
            act = batch["act"].to(device, non_blocking=True)
            mask = batch["act_mask"].to(device, non_blocking=True)
            act = act.clamp(-1, 1) * mask

            frames = obs_u8[:, :-1].float() / 255.0
            actions = torch.zeros_like(act)
            actions[:, 1:] = act[:, :-1]
            act_mask = torch.zeros_like(mask)
            act_mask[:, 1:] = mask[:, :-1]
        else:
            frames = batch["obs"].to(device, non_blocking=True)
            if frames.dtype == torch.uint8:
                frames = frames.float() / 255.0
            actions = None
            act_mask = None
        
        B = frames.shape[0]
        
        # Encode frames
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            patches = temporal_patchify(frames, tokenizer.num_patches)
            z_btLd, _ = tokenizer.encoder(patches)
            z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=dyn.n_spatial, k=packing_factor)
            
            T = z1.shape[1]
            k_max = dyn.k_max
            emax = int(round(math.log2(k_max)))
            
            # Use finest step (d_min) for clean evaluation
            step_idx = torch.full((B, T), emax, device=device, dtype=torch.long)
            
            # Sample random tau for each position
            K = k_max
            u = torch.rand((B, T), device=device, dtype=torch.float32)
            j_idx = torch.floor(u * K).to(torch.long)
            sigma = j_idx.to(torch.float32) / K
            sigma_idx = j_idx
            
            # Corrupt inputs
            z0 = torch.randn_like(z1)
            z_tilde = (1.0 - sigma)[..., None, None] * z0 + sigma[..., None, None] * z1
            
            # Forward
            z1_hat, _ = dyn(actions, step_idx, sigma_idx, z_tilde, act_mask=act_mask, agent_tokens=None)
            
            # Compute MSE
            flow_mse = (z1_hat.float() - z1.float()).pow(2).mean()
        
        total_flow_mse += flow_mse.item() * B
        total_samples += B
    
    if dyn_was_training:
        dyn.train()
    
    avg_flow_mse = total_flow_mse / max(1, total_samples)
    
    return {
        "val/flow_mse": avg_flow_mse,
        "val/samples": total_samples,
    }


@torch.no_grad()
def log_viz(*, gt: torch.Tensor, pred: torch.Tensor, ctx_length: int, step: int, max_items: int = 4, gap_px: int = 16):
    """Log visualization to wandb."""
    B, T, C, H, W = gt.shape
    Bv = min(B, max_items)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        x = x[:Bv]
        B_, T_, C_, H_, W_ = x.shape
        ctx = int(max(0, min(ctx_length, T_)))
        y = x.permute(0, 2, 3, 1, 4).contiguous().view(B_, C_, H_, T_ * W_)
        if gap_px > 0 and 0 < ctx < T_:
            split = ctx * W_
            gap = torch.zeros((B_, C_, H_, gap_px), device=y.device, dtype=y.dtype)
            y = torch.cat([y[..., :split], gap, y[..., split:]], dim=-1)
        return y

    gt_t = tile_time(gt)
    pr_t = tile_time(pred)

    panel = torch.cat([gt_t, pr_t], dim=2)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)
    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    big_hwc = big.permute(1, 2, 0).cpu().numpy()

    wandb.log({"eval/viz": wandb.Image(big_hwc, caption=f"rows=GT/Pred | ctx={ctx_length} | T={T}")}, step=step)


def train(args):
    conf = MiniConf.load(args.config)

    if is_rank0():
        conf.pprint()

    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(conf.get("seed", int) + rank)

    # ---- data ----
    dataset = WMDataset(**conf.select("data"))
    
    # Split by trajectory (respects episode boundaries) - same as tokenizer
    try:
        val_fraction = conf.get("data/val_fraction", float)
    except KeyError:
        val_fraction = 0.1
    train_dataset, val_dataset = split_by_trajectory(dataset, val_fraction=val_fraction, seed=conf.get("seed", int))
    
    if is_rank0():
        print(f"[Data] Train: {len(train_dataset):,} sequences, Val: {len(val_dataset):,} sequences")

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None

    loader = DataLoader(
        train_dataset,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
        shuffle=(sampler is None),
        collate_fn=collate_batch,
        **conf.get("dynamics/dataloader")
    )
    
    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_batch,
        **conf.get("dynamics/dataloader")
    )

    # ---- tokenizer (frozen) ----
    tokenizer_ckpt = conf.get("tokenizer/checkpoint_dir", str) + "/latest.pt"
    tokenizer = load_frozen_tokenizer(tokenizer_ckpt, device=device)

    H = tokenizer.H
    W = tokenizer.W
    C = tokenizer.C
    patch = tokenizer.num_patches
    packing_factor = conf.get("dynamics/packing_factor", int)

    # ---- model ----
    ckpt_dir = Path(conf.get("dynamics/training/ckpt_dir", str))
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize model (or resume from checkpoint)
    step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if is_rank0():
            print(f"[Resume] Loading checkpoint from {args.resume}")
        model, ckpt_info = Dynamics.from_checkpoint(Path(args.resume), tokenizer=tokenizer, device=str(device))
        model = model.to(device)
        step = ckpt_info["step"]
        start_epoch = ckpt_info["epoch"]
        best_val_loss = ckpt_info.get("best_val_loss", float('inf'))
        if is_rank0():
            print(f"[Resume] Resumed from step {step}, epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    else:
        model = Dynamics(tokenizer=tokenizer, device=str(device), **conf.select("dynamics")).to(device)

    if is_rank0():
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Learnable parameters: {param_count:,}")

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    # ---- wandb ----
    if is_rank0():
        wandb.init(
            **conf.get("dynamics/wandb"),
            config=conf.asdict(),
            resume="allow" if args.resume else None,
        )

    # ---- train ----
    model.train()
    t0 = time.monotonic()
    max_steps = conf.get("dynamics/optim/max_steps")
    log_every = conf.get("dynamics/training/log_every")
    print_every = conf.get("dynamics/training/print_every")
    save_every = conf.get("dynamics/training/save_every")
    eval_every = conf.get("dynamics/eval/every")
    grad_accum = conf.get("dynamics/optim/grad_accum")
    k_max = conf.get("dynamics/k_max", int)
    use_actions = conf.get("dynamics/use_actions", bool)
    
    # Validation config with defaults
    try:
        val_every = conf.get("dynamics/training/val_every", int)
    except KeyError:
        val_every = 1000
    try:
        val_max_batches = conf.get("dynamics/training/val_max_batches", int)
    except KeyError:
        val_max_batches = 25

    accum_step = 0
    step_t0 = time.monotonic()

    # Get the actual model (unwrap DDP if needed)
    model_module = model.module if hasattr(model, "module") else model

    while step < max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for batch in loader:
                # ---- prepare data ----
                if use_actions and "act" in batch:
                    obs_u8 = batch["obs"].to(device, non_blocking=True)
                    act = batch["act"].to(device, non_blocking=True)
                    mask = batch["act_mask"].to(device, non_blocking=True)
                    act = act.clamp(-1, 1) * mask

                    frames = obs_u8[:, :-1].float() / 255.0
                    actions = torch.zeros_like(act)
                    actions[:, 1:] = act[:, :-1]
                    act_mask = torch.zeros_like(mask)
                    act_mask[:, 1:] = mask[:, :-1]
                else:
                    frames = batch["obs"].to(device, non_blocking=True)
                    if frames.dtype == torch.uint8:
                        frames = frames.float() / 255.0
                    actions = None
                    act_mask = None

                # ---- train step ----
                accumulate = (accum_step + 1) % grad_accum != 0
                aux = model_module.train_step(
                    frames=frames,
                    actions=actions,
                    act_mask=act_mask,
                    step=step,
                    accumulate=accumulate,
                )
                loss = aux["loss"]
                accum_step += 1

                if not accumulate:
                    step += 1
                    # Step the scheduler
                    model_module.scheduler.step()
                else:
                    continue

                step_time = time.monotonic() - step_t0
                step_t0 = time.monotonic()

                # ---- logging ----
                if is_rank0() and (step % log_every == 0):
                    wandb.log({
                        "loss/total": float(loss.item()),
                        "loss/flow_mse": float(aux["flow_mse"].item()),
                        "loss/bootstrap_mse": float(aux["bootstrap_mse"].item()),
                        "loss/emp": float(aux["loss_emp"].item()),
                        "loss/self": float(aux["loss_self"].item()),
                        "stats/sigma_mean": float(aux["sigma_mean"].item()),
                        "lr": float(model_module.opt.param_groups[0]["lr"]),
                        "time/hrs": (time.monotonic() - t0) / 3600.0,
                        "time/step_ms": step_time * 1000.0,
                        "time/samples_per_sec": frames.shape[0] * grad_accum / step_time,
                    }, step=step)

                if is_rank0() and (step % print_every == 0):
                    lr = model_module.opt.param_groups[0]["lr"]
                    print(
                        f"step {step:07d} | loss={loss.item():.6f} "
                        f"| flow_mse={aux['flow_mse'].item():.6f} | boot_mse={aux['bootstrap_mse'].item():.6f} "
                        f"| lr={lr:.2e} | {step_time*1000:.1f}ms/step"
                    )

                # ---- validation ----
                if is_rank0() and val_every > 0 and (step % val_every == 0):
                    torch.cuda.empty_cache()
                    
                    val_metrics = run_validation_dynamics(
                        dyn=model_module,
                        tokenizer=tokenizer,
                        val_loader=val_loader,
                        device=device,
                        packing_factor=packing_factor,
                        use_actions=use_actions,
                        step=step,
                        max_batches=val_max_batches,
                    )
                    
                    wandb.log(val_metrics, step=step)
                    print(
                        f"step {step:07d} | VAL flow_mse={val_metrics['val/flow_mse']:.6f}"
                    )
                    
                    # Best model checkpointing
                    if val_metrics['val/flow_mse'] < best_val_loss:
                        best_val_loss = val_metrics['val/flow_mse']
                        model_module.save_checkpoint(ckpt_dir / "best.pt", step, epoch, best_val_loss, full_config=conf.asdict())
                        print(f"  -> New best model saved (val_flow_mse={best_val_loss:.6f})")
                    
                    torch.cuda.empty_cache()

                # ---- eval (sampling) ----
                if is_rank0() and eval_every > 0 and (step % eval_every == 0):
                    torch.cuda.empty_cache()

                    B_eval = min(frames.shape[0], conf.get("dynamics/eval/batch_size", int))
                    sched = make_tau_schedule(
                        k_max=k_max,
                        schedule=conf.get("dynamics/eval/schedule", str),
                        d=conf.get("dynamics/eval/d", float)
                    )

                    run_eval(
                        tokenizer=tokenizer,
                        dyn=model_module,
                        frames=frames[:B_eval],
                        actions=actions[:B_eval] if actions is not None else None,
                        act_mask=None,
                        H=H, W=W, C=C, patch=patch,
                        packing_factor=packing_factor,
                        k_max=k_max,
                        ctx_length=conf.get("dynamics/eval/ctx", int),
                        horizon=conf.get("dynamics/eval/horizon", int),
                        sched=sched,
                        max_items=conf.get("dynamics/eval/max_items", int),
                        step=step,
                    )

                    torch.cuda.empty_cache()

                # ---- ckpt ----
                if is_rank0() and save_every > 0 and (step % save_every == 0):
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    model_module.save_checkpoint(ckpt_path, step, epoch, best_val_loss, full_config=conf.asdict())
                    model_module.save_checkpoint(ckpt_dir / "latest.pt", step, epoch, best_val_loss, full_config=conf.asdict())

                if step >= max_steps:
                    break

            start_epoch = epoch + 1

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/project.yaml")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    train(p.parse_args())
