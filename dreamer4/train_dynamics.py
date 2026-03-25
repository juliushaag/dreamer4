# train_dynamics.py
from collections import defaultdict
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

from dreamer4.datasets.dataset_utils import load_datasets
from miniconf import MiniConf
from datasets.robocasa_dataset import RoboCasaDataset, split_by_trajectory as robocasa_split
from dreamer4.datasets.dmc_dataset import DMCDataset, collate_batch, split_by_trajectory as dmc_split
from model import (
    Tokenizer,
    Dynamics,
    _emax_from_kmax,
    _sample_step_excluding_dmin,
    _sample_tau_for_step,
    temporal_patchify, temporal_unpatchify,
    pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

from dreamer4.train_utils import TrainingState, create_cosine_scheduler, init_distributed, is_rank0, num_model_params, seed_everything, worker_init_fn

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
    
    if is_rank0():
        print(f"[Data] Train: {len(train_dataset):,} sequences, Val: {len(val_dataset):,} sequences")

    train_dataset, val_dataset = load_datasets(conf.get("dataset"))
    
    if is_rank0():
        print(f"[Data] Train: {len(train_dataset):,} sequences, Val: {len(val_dataset):,} sequences")

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler=None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn,
        shuffle=(train_sampler is None),
        collate_fn=collate_batch,
        **conf.get("tokenizer/dataloader")
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_batch,
        **conf.get("tokenizer/dataloader")
    )   

    # ---- tokenizer (frozen) ----
    H = conf.get(f"dataset/image_height", int)
    W = conf.get(f"dataset/image_width", int)
    C = 3  # RGB
    P = conf.get("tokenizer/patch_size", int)
    
    assert H % P == 0 and W % P == 0
   
    tokenizer_state = TrainingState(
        model=Tokenizer(C, H, W)
    ) 

    tokenizer_ckpt = "logs/tokenizer_long_run/best.pt"
    tokenizer_state.load(tokenizer_ckpt)

    tokenizer : Tokenizer = tokenizer_state.model
    patch = tokenizer.n_patches

    if is_rank0():
        print(f"Loaded tokenizer checkpoint {tokenizer_ckpt}: steps={tokenizer_state.steps}, best_val={tokenizer_state.best_val}")

    packing_factor = conf.get("dynamics/packing_factor", int)

    # ---- model ----
    ckpt_dir = Path(conf.get("dynamics/training/ckpt_dir", str))
    os.makedirs(ckpt_dir, exist_ok=True)

    
    state = TrainingState(
        Dynamics(tokenizer=tokenizer, device=device)
    )

     # Create optimizer (like Tokenizer)
    state.opt = torch.optim.AdamW(
        state.model.parameters(),
        fused=torch.cuda.is_available(),
        **conf.get("dynamics/optim")
    )

    # Learning rate scheduler with warmup + cosine decay
    state.scheduler = create_cosine_scheduler(
        optimizer=state.opt,
        **conf.get("dynamics/opt_sched"),
        max_steps=conf.get("dynamics/training/maxsteps"),
        base_lr=conf.get("dynamics/optim/lr"),
    )

    if conf.get("compile", bool):
        state.model.forward = torch.compile(state.model.forward)


    if args.resume:
        
        state.load(args.resume)

        if is_rank0():
            print(f"[Resume] Resumed from step {state.steps}, best_val_loss={state.best_val:.6f}")
  
    if is_rank0():
        params = num_model_params(state.model)
        print(f"Learnable parameters: {params["trainable"]}, all: {params["all"]}")

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    # ---- wandb ----
    if is_rank0():
        ckpt_dir = Path(conf.get("dynamics/training/ckpt_dir", str))
        os.makedirs(ckpt_dir, exist_ok=True)

        run = wandb.init(
            **conf.get("dynamics/wandb"),
            config=conf.asdict(),
            id=state.wandb_run,
            resume="allow" if args.resume else None,
        )
        state.wandb_run = run.id
    
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
    val_every = conf.get("dynamics/training/val_every", int)
 
    val_max_batches = conf.get("dynamics/training/val_max_batches", int)
 
    step_t0 = time.monotonic()


    packing_factor: int = conf.get("dynamics/packing_factor")
    self_fraction: float = conf.get("dynamics/self_fraction")
    bootstrap_start: int = conf.get("dynamics/bootstrap_start")
   
    
    model.train()
    start_step = state.steps
    for state.steps in range(start_step, max_steps):
        
        step_t0 = time.monotonic()        

        aux = defaultdict(default_factory=lambda: torch.zeros((grad_accum)))

        model.train()
        state.opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", torch.bfloat16):
            for gs in range(grad_accum):
                batch = next(train_loader)
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
            with torch.no_grad():
                patches = temporal_patchify(frames, tokenizer.n_patches)
                z_btLd, _ = tokenizer.encoder(patches)
                z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=model.n_spatial, k=model.packing_factor)

            device = z1.device
            B, T = z1.shape[:2]
            B_self = int(round(self_fraction * B))
            B_self = max(0, min(B - 1, B_self))
            B_emp = B - B_self
            emax = _emax_from_kmax(k_max)

            # action mask slices
            act_mask_full = act_mask
            act_mask_self = None if act_mask_full is None else act_mask_full[B_emp:]

            # step idx: empirical rows are finest (d_min), self rows sample coarser
            step_idx_emp = torch.full((B_emp, T), emax, device=device, dtype=torch.long)
            if B_self > 0:
                d_self, step_idx_self = _sample_step_excluding_dmin(device, B_self, T, k_max)
                step_idx_full = torch.cat([step_idx_emp, step_idx_self], dim=0)
            else:
                d_self = torch.zeros((0, T), device=device, dtype=torch.float32)
                step_idx_self = torch.zeros((0, T), device=device, dtype=torch.long)
                step_idx_full = step_idx_emp

            # sigma/tau per row/time
            sigma_full, sigma_idx_full = _sample_tau_for_step(device, B, T, k_max, step_idx_full)
            sigma_emp = sigma_full[:B_emp]
            sigma_self = sigma_full[B_emp:]
            sigma_idx_self = sigma_idx_full[B_emp:]

            # Corrupt inputs
            z0_full = torch.randn_like(z1)
            z_tilde_full = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[..., None, None] * z1
            z_tilde_self = z_tilde_full[B_emp:]

            # Weights (0.9 * sigma + 0.1 gives higher weight to higher noise levels)
            w_emp = 0.9 * sigma_emp + 0.1
            w_self = 0.9 * sigma_self + 0.1

            # Main forward
            z1_hat_full, _ = model(actions, step_idx_full, sigma_idx_full, z_tilde_full, act_mask=act_mask_full, agent_tokens=None)
            z1_hat_emp = z1_hat_full[:B_emp]
            z1_hat_self = z1_hat_full[B_emp:]

            flow_per = (z1_hat_emp.float() - z1[:B_emp].float()).pow(2).mean(dim=(2, 3))  # (B_emp,T)
            loss_emp = (flow_per * w_emp).mean()

            boot_mse = torch.zeros((), device=device, dtype=torch.float32)
            loss_self = torch.zeros((), device=device, dtype=torch.float32)

            do_boot = (B_self > 0) and (step >= bootstrap_start)
            if do_boot:
                d_half = d_self / 2.0
                step_idx_half = step_idx_self + 1
                sigma_plus = sigma_self + d_half
                sigma_idx_plus = sigma_idx_self + (torch.tensor(k_max, device=device, dtype=torch.float32) * d_half).to(torch.long)

                z1_hat_half1, _ = model(actions[B_emp:] if actions is not None else None, step_idx_half, sigma_idx_self, z_tilde_self, act_mask=act_mask_self, agent_tokens=None)
                b_prime = (z1_hat_half1.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
                z_prime = z_tilde_self.float() + b_prime * d_half[..., None, None]

                z1_hat_half2, _ = model(actions[B_emp:] if actions is not None else None, step_idx_half, sigma_idx_plus, z_prime.to(z_tilde_self.dtype), act_mask=act_mask_self, agent_tokens=None)
                b_doubleprime = (z1_hat_half2.float() - z_prime.float()) / (1.0 - sigma_plus).clamp_min(1e-6)[..., None, None]

                vhat_sigma = (z1_hat_self.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
                vbar_target = ((b_prime + b_doubleprime) / 2.0).detach()

                boot_per = (1.0 - sigma_self).pow(2) * (vhat_sigma - vbar_target).pow(2).mean(dim=(2, 3))  # (B_self,T)
                loss_self = (boot_per * w_self).mean()
                boot_mse = boot_per.mean()

    

            # Combine losses
            loss = (((loss_emp * (B - B_self)) + (loss_self * B_self)) / B) / grad_accum
            loss.backward()

            aux["loss_emp"][gs] = loss_emp.detach()
            aux["loss_self"][gs] = loss_self.detach()
            aux["bootstrap_mse"][gs] = boot_mse.detach()
            aux["flow_mse"][gs] = flow_per.detach().mean()
            aux["loss"][gs] = loss.mean()
            aux["sigma_mean"][gs] = sigma_full.mean().mean()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        state.opt.step()
        state.scheduler.step()               

        for k in aux:
            aux[k] = aux[k].mean()

        step_time = time.monotonic() - step_t0
        step_t0 = time.monotonic()

        # ---- logging ----
        if is_rank0() and (step % log_every == 0):
            wandb.log({
                "loss/total": float(aux["loss"].item()),
                "loss/flow_mse": float(aux["flow_mse"].item()),
                "loss/bootstrap_mse": float(aux["bootstrap_mse"].item()),
                "loss/emp": float(aux["loss_emp"].item()),
                "loss/self": float(aux["loss_self"].item()),
                "stats/sigma_mean": float(aux["sigma_mean"].item()),
                "lr": float(state.opt.param_groups[0]["lr"]),
                "time/hrs": (time.monotonic() - t0) / 3600.0,
                "time/step_ms": step_time * 1000.0,
                "time/samples_per_sec": frames.shape[0] * grad_accum / step_time,
            }, step=step)

        if is_rank0() and (step % print_every == 0):
            lr = state.opt.param_groups[0]["lr"]
            print(
                f"step {step:07d} | loss={aux["loss"].item():.6f} "
                f"| flow_mse={aux['flow_mse'].item():.6f} | boot_mse={aux['bootstrap_mse'].item():.6f} "
                f"| lr={lr:.2e} | {step_time*1000:.1f}ms/step"
            )

        # ---- validation ----
        if is_rank0() and val_every > 0 and (step % val_every == 0):
            torch.cuda.empty_cache()
            
            val_metrics = run_validation_dynamics(
                dyn=model,
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
            if val_metrics['val/flow_mse'] < state.best_val:
                state.best_val = val_metrics['val/flow_mse']
                state.save(ckpt_dir / "best.pt")
                print(f"  -> New best model saved (val_flow_mse={state.best_val:.6f})")
            
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
                    dyn=model,
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

                state.save(ckpt_path)
                state.save(ckpt_dir / "latest.pt")


    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/project.yaml")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    train(p.parse_args())
