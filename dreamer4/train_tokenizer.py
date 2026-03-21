# train_tokenizer.py
import os
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import einops

import wandb

from datasets.robocasa_dataset import RoboCasaDataset, split_by_trajectory as robocasa_split
from datasets.wm_dataset import DMCDataset, collate_batch, split_by_trajectory as dmc_split
from model import (
    Tokenizer,
    temporal_patchify, temporal_unpatchify,
    lpips_on_mae_recon,
)
from miniconf import MiniConf

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
def run_validation(model, val_loader, device, P, H, W, C, max_batches=None, compute_lpips=True, lpips_frac=0.25):
    """
    Run validation and return average metrics.
    
    Args:
        model: The tokenizer model (may be DDP wrapped)
        val_loader: Validation data loader
        device: Device to run on
        P: Patch size
        H, W, C: Image dimensions
        max_batches: Maximum number of batches to evaluate (None = all)
        compute_lpips: Whether to compute LPIPS metric
        lpips_frac: Fraction of samples to use for LPIPS (for efficiency)
    
    Returns:
        Dictionary of average metrics
    """
    model_module = model.module if hasattr(model, "module") else model
    was_training = model_module.training
    model_module.eval()
    
    total_mse = 0.0
    total_lpips = 0.0
    total_samples = 0
    
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
            
        x = batch["obs"].to(device, non_blocking=True)
        
        # Convert uint8 to float if needed
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0
        
        B = x.shape[0]
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            patches = temporal_patchify(x, P)
            pred, mae_mask, _ = model_module(patches)
            
            # Compute MSE on masked patches only (reconstruction quality)
            diff_sq = (pred - patches).pow(2)
            diff_sq = diff_sq * mae_mask
            denom = mae_mask.sum().float().clamp_min(1.0) * diff_sq.shape[-1]
            mse = diff_sq.sum() / denom
            
            # Compute LPIPS if enabled
            if compute_lpips:
                lpips_val = lpips_on_mae_recon(
                    model_module.lpips, pred, patches, mae_mask,
                    H=H, W=W, C=C, patch=P,
                    subsample_frac=lpips_frac
                )
                total_lpips += lpips_val.item() * B
        
        total_mse += mse.item() * B
        total_samples += B
    
    if was_training:
        model_module.train()
    
    avg_mse = total_mse / max(1, total_samples)
    avg_psnr = 10.0 * np.log10(1.0 / max(avg_mse, 1e-10))
    
    metrics = {
        "val/mse": avg_mse,
        "val/psnr": avg_psnr,
        "val/samples": total_samples,
    }
    
    if compute_lpips:
        metrics["val/lpips"] = total_lpips / max(1, total_samples)
    
    return metrics


def train(args):
    conf = MiniConf.load(args.config)

    if is_rank0():
        conf.pprint()

    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(conf.get("seed", int) + rank)

    # ---- data ----
    ds_name = conf.get("dataset")
    if ds_name == "robocasa":
        dataset = RoboCasaDataset(**conf.select("robocasa_data"))
    elif ds_name == "dmc":
        dataset = DMCDataset(**conf.select("dmc_data"))
    else:
        raise ValueError(f"Invalid dataset specified {ds_name}")
    
    
    # Split by trajectory (respects episode boundaries)
    val_fraction = conf.get(f"{ds_name}_data/val_fraction", float)
   
    
    # Use appropriate split function based on dataset type
    print(ds_name)
    if ds_name == "robocasa":
        train_dataset, val_dataset = robocasa_split(dataset, val_fraction=val_fraction, seed=conf.get("seed", int))
    else:
        train_dataset, val_dataset = dmc_split(dataset, val_fraction=val_fraction, seed=conf.get("seed", int))
    
    if is_rank0():
        print(f"[Data] Train: {len(train_dataset):,} sequences, Val: {len(val_dataset):,} sequences")

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None

    loader = DataLoader(
        train_dataset,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
        shuffle=(sampler is None),
        collate_fn=collate_batch,
        **conf.get("tokenizer/dataloader")
    )

    # Validation loader (no distributed sampler, only run on rank 0)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_batch,
        **conf.get("tokenizer/dataloader")
    )

    # ---- model ----
    # Get image dimensions from the appropriate data config
    data_config_key = f"{ds_name}_data" if ds_name != "dmc" else "dmc_data"
    H = conf.get(f"{data_config_key}/image_height", int)
    W = conf.get(f"{data_config_key}/image_width", int)
    C = 3  # RGB
    P = conf.get("tokenizer/patch_size", int)
    
    assert H % P == 0 and W % P == 0
 
    ckpt_dir = Path(conf.get("tokenizer/training/ckpt_dir", str))
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Initialize model (or resume from checkpoint)
    step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if is_rank0():
            print(f"[Resume] Loading checkpoint from {args.resume}")
        model, ckpt_info = Tokenizer.from_checkpoint(Path(args.resume), device=str(device))
        model = model.to(device)
        step = ckpt_info["step"]
        start_epoch = ckpt_info["epoch"]
        best_val_loss = ckpt_info.get("best_val_loss", float('inf'))
        if is_rank0():
            print(f"[Resume] Resumed from step {step}, epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    else:
        # Map "data" namespace to the appropriate data config based on dataset
        data_ns_path = f"/{data_config_key}"
        model = Tokenizer(device=str(device), **conf.select("tokenizer", data=data_ns_path)).to(device)

    if is_rank0():
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Learnable parameters: {param_count:,}")

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    # Get model module (unwrap DDP if needed)
    model_module = model.module if hasattr(model, "module") else model

    # ---- wandb ----
    if is_rank0():
        wandb.init(
            **conf.get("tokenizer/wandb"),
            config=conf.asdict(),
            resume="allow" if args.resume else None,
        )

    # ---- train ----
    model.train()
    t0 = time.monotonic()
    max_steps = conf.get("tokenizer/training/maxsteps")
    log_every = conf.get("tokenizer/training/log_every")
    print_every = conf.get("tokenizer/training/print_every")
    save_every = conf.get("tokenizer/training/save_every")
    viz_every = conf.get("tokenizer/training/viz_every")
    viz_max_items = conf.get("tokenizer/training/viz_max_items")
    viz_max_T = conf.get("tokenizer/training/viz_max_T")
    
    # Validation config with defaults
    val_every = conf.get("tokenizer/training/val_every", int)

    val_max_batches = conf.get("tokenizer/training/val_max_batches", int)
    val_lpips_frac = conf.get("tokenizer/training/val_lpips_frac", float)
        
    grad_accum = conf.get("tokenizer/optim/grad_accum")
    
    accum_step = 0
    step_t0 = time.monotonic()
    
    while step < max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for batch in loader:
                x = batch["obs"].to(device, non_blocking=True)
                
                # Gradient accumulation: only step optimizer every grad_accum batches
                accumulate = (accum_step + 1) % grad_accum != 0
                loss, mse, lp, keep_prob, mae_mask = model.train_step(x, accumulate=accumulate)
                accum_step += 1
                
                # Only count as a "step" when we actually update weights
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
                    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                    wandb.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/mse": float(mse.item()),
                            "train/lpips": float(lp.item()),
                            "train/psnr": float(psnr.item()),
                            "stats/keep_prob": float(keep_prob.mean().item()),
                            "stats/masked_frac": float(mae_mask.float().mean().item()),
                            "lr": float(model_module.opt.param_groups[0]["lr"]),
                            "time/hrs": (time.monotonic() - t0) / 3600.0,
                            "time/step_ms": step_time * 1000.0,
                            "time/samples_per_sec": x.shape[0] * grad_accum / step_time,
                        },
                        step=step,
                    )

                if is_rank0() and (step % print_every == 0):
                    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                    lr = model_module.opt.param_groups[0]["lr"]
                    print(
                        f"step {step:07d} | loss={loss.item():.6f} "
                        f"| mse={mse.item():.6f} | lpips={lp.item():.4f} "
                        f"| psnr={psnr.item():.2f} | keep={keep_prob.mean().item():.3f} "
                        f"| lr={lr:.2e} | {step_time*1000:.1f}ms/step"
                    )

                # ---- validation ----
                if is_rank0() and val_every > 0 and (step % val_every == 0):
                    torch.cuda.empty_cache()
                    
                    val_metrics = run_validation(
                        model, val_loader, device, P, H, W, C,
                        max_batches=val_max_batches,
                        compute_lpips=True,
                        lpips_frac=val_lpips_frac,
                    )
                    
                    wandb.log(val_metrics, step=step)
                    print(
                        f"step {step:07d} | VAL mse={val_metrics['val/mse']:.6f} "
                        f"| psnr={val_metrics['val/psnr']:.2f} "
                        f"| lpips={val_metrics.get('val/lpips', 0):.4f}"
                    )
                    
                    # Best model checkpointing
                    if val_metrics['val/mse'] < best_val_loss:
                        best_val_loss = val_metrics['val/mse']
                        model_module.save_checkpoint(ckpt_dir / "best.pt", step, epoch, best_val_loss, full_config=conf.asdict())
                        print(f"  -> New best model saved (val_mse={best_val_loss:.6f})")
                    
                    torch.cuda.empty_cache()

                # ---- viz ----
                if is_rank0() and viz_every > 0 and (step % viz_every == 0):
                    torch.cuda.empty_cache()

                    B, T, C, H, W = x.shape
                    Tv = min(T, viz_max_T)
                    Bv = min(B, viz_max_items)

                    x_viz = x[:Bv, :Tv]
                    # Convert uint8 to float if needed
                    if x_viz.dtype == torch.uint8:
                        x_viz = x_viz.to(torch.float32) / 255.0
                    
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            patches = temporal_patchify(x_viz, P)
                            pred, mae_mask, _ = model_module(patches)

                            masked_input_btnd = torch.where(mae_mask, torch.zeros_like(patches), patches)
                            recon_masked_btnd = torch.where(mae_mask, pred, patches)

                            def to_tiled_image(x_btnd: torch.Tensor) -> torch.Tensor:
                                """(B,T,Np,Dp) patches -> (B,C,H,T*W) tiled image"""
                                imgs = temporal_unpatchify(x_btnd, H, W, C, P)
                                return einops.rearrange(imgs, 'b t c h w -> b c h (t w)')

                            panels = [to_tiled_image(p) for p in [patches, masked_input_btnd, recon_masked_btnd, pred]]
                            panel = torch.cat(panels, dim=2)
                            
                            big = einops.rearrange(panel, 'b c h w -> (b h) w c')
                            big = (big.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()

                            wandb.log({
                                "viz/reconstruction": wandb.Image(big, caption="rows=target/masked/recon_masked/recon_full"),
                            }, step=step)
                                        
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
