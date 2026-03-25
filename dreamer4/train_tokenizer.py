# train_tokenizer.py
from collections import defaultdict
import itertools
import os
import time
import argparse
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import einops

import wandb
from datasets import dmc_dataset, robocasa_dataset
from datasets.dataset_utils import load_datasets, collate_batch
from train_utils import TrainingState, create_cosine_scheduler, init_distributed, is_rank0, num_model_params, seed_everything, worker_init_fn
from model import (
    Tokenizer,
    temporal_patchify, temporal_unpatchify,
)
from miniconf import MiniConf

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)


def recon_loss_from_mae(pred : torch.Tensor, target : torch.Tensor, mask : torch.Tensor):
    # Compute masked squared error without cloning pred
    diff_sq = (pred - target).square_()
    diff_sq.mul_(mask)

    denom = mask.sum().float().clamp_min_(1.0) * diff_sq.shape[-1]

    return diff_sq.sum(dtype=torch.float32) / denom


def lpips_on_mae_recon(
    lpips_fn,
    pred_btnd: torch.Tensor,
    target_btnd: torch.Tensor,
    mae_mask_btNp1: torch.Tensor,
    *,
    H: int, W: int, C: int, patch: int,
    subsample_frac: float = 1.0,
) -> torch.Tensor:
    
    recon_masked_btnd = torch.where(mae_mask_btNp1, pred_btnd, target_btnd)
    recon = temporal_unpatchify(recon_masked_btnd,  H, W, C, patch)
    tgt   = temporal_unpatchify(target_btnd,        H, W, C, patch)

    if subsample_frac < 1.0:
        step = max(1, int(1.0 / subsample_frac))
        recon = recon[:, ::step]
        tgt   = tgt[:, ::step]

    recon = (recon.clamp(0, 1) * 2.0 - 1.0)
    tgt   = (tgt.clamp(0, 1)   * 2.0 - 1.0)

    recon = einops.rearrange(recon, 'b t c h w -> (b t) c h w')
    tgt   = einops.rearrange(tgt,   'b t c h w -> (b t) c h w')
    
    lp = lpips_fn(recon, tgt)

    return lp.mean()


@torch.inference_mode()
def run_validation(model : torch.nn.Module, val_loader : torch.utils.data.DataLoader, device, P, H, W, C, lpips, lpips_frac=0.25, ddp=False, world_size=None):
    model.eval()
    max_steps = 100
    total_mse = torch.zeros(max_steps)
    total_lpips = torch.zeros(max_steps)
        
    if not ddp: world_size = 1

    for i, batch in enumerate(val_loader):

        x = batch["obs"].to(device, non_blocking=True)
        
        # Convert uint8 to float if needed
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0
        
        B = x.shape[0]
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            patches = temporal_patchify(x, P)
            pred, mae_mask, _ = model(patches)
            
            diff_sq = (pred - patches).pow(2) * mae_mask

            denom = mae_mask.sum().float().clamp_min(1.0) * diff_sq.shape[-1]
            
            mse = diff_sq.sum() / denom
            
            lpips_val = lpips_on_mae_recon(
                lpips, pred, patches, mae_mask,
                H=H, W=W, C=C, patch=P,
                subsample_frac=lpips_frac
            )
    
        total_lpips[i] = lpips_val * B
        total_mse[i] = mse * B

        if i >= max_steps - 1:
            break

    if ddp:
        dist.all_reduce(total_mse, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_lpips, op=dist.ReduceOp.SUM)

        total_mse = total_mse / world_size
        total_lpips = total_lpips / world_size

    avg_mse = total_mse.mean().item()
    avg_lpips = total_lpips.mean().item()
       
    avg_psnr = 10.0 * np.log10(1.0 / max(avg_mse, 1e-10))

    metrics = {
        "val/mse": avg_mse,
        "val/psnr": avg_psnr,
        "val/samples": max_steps * world_size,
        "val/lpips" : avg_lpips,
    }
    
    return metrics

@torch.inference_mode()
def run_visualization(model : torch.nn.Module, x, P, viz_max_T : int, viz_max_items : int) -> np.ndarray: 
    B, T, C, H, W = x.shape
    Tv = min(T, viz_max_T)
    Bv = min(B, viz_max_items)

    x_viz = x[:Bv, :Tv]
    # Convert uint8 to float if needed
    if x_viz.dtype == torch.uint8:
        x_viz = x_viz.to(torch.float32) / 255.0
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patches = temporal_patchify(x_viz, P)
        pred, mae_mask, _ = model(patches)

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

    return big


def train(args):

    conf = MiniConf.load(args.config)

    if is_rank0():
        conf.pprint()

    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(conf.get("seed", int) + rank)

    # 1. Load model

    # ---- model ----
    # Get image dimensions from the appropriate data config
    H = conf.get(f"dataset/image_height", int)
    W = conf.get(f"dataset/image_width", int)
    C = 3  # RGB
    P = conf.get("tokenizer/patch_size", int)
    
    assert H % P == 0 and W % P == 0
   
   
    model = Tokenizer(C, H, W, **conf.select("tokenizer")).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), 
        fused=torch.cuda.is_available(),
        **conf.get("tokenizer/optim")
    )
    
    scheduler = create_cosine_scheduler(
        optimizer=opt,
        **conf.get("tokenizer/opt_sched"),
        max_steps=conf.get("tokenizer/training/maxsteps"),
        base_lr=conf.get("tokenizer/optim/lr"),
    )

    state = TrainingState(
        opt=opt, model=model, scheduler=scheduler, conf=conf
    )

    if args.resume is not None:
        if is_rank0():
            print(f"[Resume] Loading checkpoint from {args.resume}")

        state.load(args.resume)

        if is_rank0():
            print(f"[Resume] Resumed from step {state.steps}, best_val_loss={state.best_val:.6f}")

        
    if is_rank0():
        params = num_model_params(state.model)
        print(f"Learnable parameters: {params['trainable'] / 1e6:.4}MB, all: {params['all'] / 1e6:.4}MB")

    if conf.get("compile"):
        model.forward = torch.compile(model.forward)

    # 2. Load dataset
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

    # 3. Lpips 
    lpips_fn : str = conf.get("tokenizer/lpips/net")
    lpips_frac : float = conf.get("tokenizer/lpips/frac")
    lpips_weight : float = conf.get("tokenizer/lpips/weight")
    
    lpips_model = lpips.LPIPS(net=lpips_fn, verbose=False).to(device)
    lpips_model.eval()
    lpips_model.requires_grad_(False)


    # ---- wandb ----
    if is_rank0():
        ckpt_dir = Path(conf.get("tokenizer/training/ckpt_dir", str))
        os.makedirs(ckpt_dir, exist_ok=True)
        
        run = wandb.init(
            **conf.get("tokenizer/wandb"),
            config=conf.asdict(),
            id=state.wandb_run,
            resume="allow" if args.resume else None,
        )

        state.wandb_run = run.id

    # ---- train ----
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

    val_lpips_frac = conf.get("tokenizer/training/val_lpips_frac", float)
        
    grad_accum = conf.get("tokenizer/training/grad_accum")
    

    train_loader = itertools.cycle(train_loader)

    
    model.train()
    start_step = state.steps
    for state.steps in range(start_step, max_steps):
        
        aux = defaultdict(lambda: torch.zeros((grad_accum,)))

        step_t0 = time.monotonic()        
    
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", torch.bfloat16):
            for gs in range(grad_accum):
                batch = next(train_loader)
                x = batch["obs"].to(device, non_blocking=True)

                patches = temporal_patchify(x, P)
                pred, mae_mask, keep_prob = model(patches)
                
                mse = recon_loss_from_mae(pred, patches, mae_mask)

                lp = lpips_on_mae_recon(
                    lpips_model, pred, patches, mae_mask,
                    H=H, W=W, C=C, patch=P,
                    subsample_frac=lpips_frac
                )

                loss : torch.Tensor = (mse + lpips_weight * lp) / grad_accum
                loss.backward()
                
                aux["mse_losses"][gs] = mse.detach()
                aux["lpips_losses"][gs] = lp.detach()
                aux["total_losses"][gs] = loss.detach()
        

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        step_time = time.monotonic() - step_t0

        for k in aux:
            aux[k] = aux[k].mean()

        # ---- validation ----
        if is_rank0() and (state.steps % val_every == 0):
            torch.cuda.empty_cache()
            
            val_metrics = run_validation(
                model, val_loader, device, P, H, W, C, lpips_model,
                lpips_frac=val_lpips_frac,
            )
            
            wandb.log(val_metrics, step=state.steps)
            print(
                f"step {state.steps:07d} " 
                f"| VAL mse={val_metrics['val/mse']:.6f} "
                f"| psnr={val_metrics['val/psnr']:.2f} "
                f"| lpips={val_metrics['val/lpips']:.4f}"
            )
            
            # Best model checkpointing
            if val_metrics['val/mse'] < state.best_val:
                state.best_val = val_metrics['val/mse']
                state.save(ckpt_dir / "best.pt")
                print(f"  -> New best model saved (val_mse={state.best_val:.6f})")
            
            torch.cuda.empty_cache()

        # ---- logging ----
        if is_rank0() and (state.steps % log_every == 0):
            psnr = 10.0 * torch.log10(1.0 / torch.clamp_min(aux["mse_losses"], 1e-10))
            wandb.log(
                {
                    **{f"train/{k}": aux[f"{k}_losses"].item() for k in ["mse", "total", "lpips"] },
                    "train/psnr": float(psnr.item()),
                    "stats/keep_prob": float(keep_prob.mean().item()),
                    "stats/masked_frac": float(mae_mask.float().mean().item()),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "time/hrs": (time.monotonic() - t0) / 3600.0,
                    "time/step_ms": step_time * 1000.0,
                    "time/samples_per_sec": x.shape[0] * grad_accum / step_time,
                },
                step=state.steps,
            )

        if is_rank0() and (state.steps % print_every == 0):
            psnr = 10.0 * torch.log10(1.0 / aux["mse_losses"].clamp_min(1e-10))
            lr = opt.param_groups[0]["lr"]
            print(
                f"step {state.steps:07d} | loss={aux['total_losses'].item():.6f} "
                f"| mse={aux['mse_losses'].item():.6f} | lpips={aux['lpips_losses'].item():.4f} "
                f"| psnr={psnr.item():.2f} | keep={keep_prob.mean().item():.3f} "
                f"| lr={lr:.2e} | {step_time*1000:.1f}ms/step"
            )

        # ---- viz ----
        if is_rank0() and viz_every > 0 and (state.steps % viz_every == 0):
            torch.cuda.empty_cache()

            image = run_visualization(model, x, P, viz_max_T, viz_max_items)
            wandb.log({
                "viz/reconstruction": wandb.Image(image, caption="rows=target/masked/recon_masked/recon_full"),
            }, step=state.steps)
                                
            torch.cuda.empty_cache()
    
        # ---- ckpt ----
        if is_rank0() and save_every > 0 and (state.steps % save_every == 0):
            ckpt_path = ckpt_dir / f"step_{state.steps:07d}.pt"
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
