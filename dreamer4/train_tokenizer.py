# train_tokenizer.py
import os
import time
import random
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler

import wandb

from task_set import TASK_SET
from sharded_frame_dataset import ShardedFrameDataset
from model import (
    Encoder, Decoder, Tokenizer,
    temporal_patchify, temporal_unpatchify,
)
from miniconf import MiniConf

import lpips


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
def log_tokenizer_viz_wandb(
    *,
    x_btchw: torch.Tensor,          # (B,T,C,H,W) float in [0,1]
    pred_btnd: torch.Tensor,        # (B,T,Np,Dp) float in [0,1]
    mae_mask_btNp1: torch.Tensor,   # (B,T,Np,1) bool True=masked
    patch: int,
    step: int,
    max_items: int = 8,
    max_T: int = 6,
    tag: str = "tokenizer/viz",
):
    B, T, C, H, W = x_btchw.shape
    Tv = min(T, max_T)
    Bv = min(B, max_items)

    # patchify target
    target_btnd = temporal_patchify(x_btchw[:, :Tv], patch)  # (B,Tv,Np,Dp)

    # panels (patch space)
    masked_input_btnd = torch.where(mae_mask_btNp1[:, :Tv], torch.zeros_like(target_btnd), target_btnd)
    recon_masked_btnd = torch.where(mae_mask_btNp1[:, :Tv], pred_btnd[:, :Tv], target_btnd)
    recon_full_btnd   = pred_btnd[:, :Tv]

    # to image space (B,T,C,H,W)
    target_img = temporal_unpatchify(target_btnd,       H, W, C, patch)
    masked_img = temporal_unpatchify(masked_input_btnd, H, W, C, patch)
    rmask_img  = temporal_unpatchify(recon_masked_btnd, H, W, C, patch)
    rfull_img  = temporal_unpatchify(recon_full_btnd,   H, W, C, patch)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        # (B,T,C,H,W) -> (B,C,H,T*W)
        x = x[:, :Tv]
        return x.permute(0, 2, 3, 1, 4).contiguous().view(x.shape[0], C, H, Tv * W)

    tgt = tile_time(target_img[:Bv])
    msk = tile_time(masked_img[:Bv])
    rm  = tile_time(rmask_img[:Bv])
    rf  = tile_time(rfull_img[:Bv])

    panel = torch.cat([tgt, msk, rm, rf], dim=2)  # (Bv,C,4H,Tv*W)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)  # (C,Bv*4H,Tv*W)

    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    big_hwc = big.permute(1, 2, 0).cpu().numpy()

    wandb.log(
        {
            tag: wandb.Image(
                big_hwc,
                caption="rows=target/masked/recon_masked/recon_full",
            ),
            "tokenizer/masked_frac": float(mae_mask_btNp1[:, :Tv].float().mean().item()),
        },
        step=step,
    )


def save_ckpt(path: Path, *, step: int, epoch: int, model, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "epoch": epoch,
        "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "opt": model.opt.state_dict(),
        "args": vars(args),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, *, model) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"]
    (model.module if hasattr(model, "module") else model).load_state_dict(state, strict=True)
    model.opt.load_state_dict(ckpt["opt"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def train(args):

    conf = MiniConf.load(args.config)

    conf.pprint()

    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(conf.get("seed", int) + rank)

    # ---- data ----
    dataset = ShardedFrameDataset(**conf.select("data"))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None

    loader = DataLoader(
        dataset,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
        shuffle=(sampler is None),
        **conf.asdict("tokenizer/dataloader")
    )

    # ---- model ----
    H = conf.get("data/image_height", int)
    W = conf.get("data/image_width", int)
    C = conf.get("data/image_channels", int)
    

    P = conf.get("tokenizer/num_patches", int)  # This is actually patch_size (kernel/stride)
    
    assert H % P == 0 and W % P == 0
    n_patches = (H // P) * (W // P)  # number of patches
    d_patch = P * P * C              # patch dimension (pixels per patch)


    ckpt_dir = conf.get("tokenizer/training/ckpt_dir", str)
    
    enc = Encoder(n_patches=n_patches, d_patch=d_patch, **conf.select("tokenizer", data="/data"))
    dec = Decoder(n_patches=n_patches, d_patch=d_patch, **conf.select("tokenizer", data="/data"))
    model = Tokenizer(enc, dec, device, **conf.select("tokenizer", data="/data")).to(device)

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
            **conf.asdict("/tokenizer/wandb"),
            config=conf.data
        )

    # ---- train ----
    model.train()
    t0 = time.monotonic()
    step = 0
    start_epoch = 0
    max_steps = conf.get("tokenizer/training/maxsteps")
    log_every = conf.get("tokenizer/training/log_every")
    print_every = conf.get("tokenizer/training/print_every")
    save_every = conf.get("tokenizer/training/save_every")
    viz_every = conf.get("tokenizer/training/viz_every")
    viz_max_items = conf.get("tokenizer/training/viz_max_items")
    viz_max_T = conf.get("tokenizer/training/viz_max_T")
    grad_accum = conf.get("tokenizer/optim/grad_accum")
    
    accum_step = 0  # Track accumulation steps
    step_t0 = time.monotonic()  # Track time per step
    
    while step < max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for x in loader:
                x = x.to(device, non_blocking=True)  # (B,T,C,H,W)
                
                # Gradient accumulation: only step optimizer every grad_accum batches
                accumulate = (accum_step + 1) % grad_accum != 0
                loss, mse, lp, keep_prob, mae_mask = model.train_step(x, accumulate=accumulate)
                accum_step += 1
                
                # Only count as a "step" when we actually update weights
                if not accumulate:
                    step += 1
                else:
                    continue  # Skip logging/viz on accumulation steps

                # Measure step time
                step_time = time.monotonic() - step_t0
                step_t0 = time.monotonic()

                # ---- logging ----
                if is_rank0() and (step % log_every == 0):
                    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                    wandb.log(
                        {
                            "loss/total": float(loss.item()),
                            "loss/mse": float(mse.item()),
                            "loss/lpips": float(lp.item()),
                            "stats/psnr": float(psnr.item()),
                            "stats/keep_prob": float(keep_prob.mean().item()),
                            "stats/masked_frac": float(mae_mask.float().mean().item()),
                            "lr": float(model.opt.param_groups[0]["lr"]),
                            "time/hrs": (time.monotonic() - t0) / 3600.0,
                            "time/step_ms": step_time * 1000.0,
                            "time/samples_per_sec": x.shape[0] * grad_accum / step_time,
                        },
                        step=step,
                    )

                if is_rank0() and (step % print_every == 0):
                    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                    print(
                        f"step {step:07d} | loss={loss.item():.6f} "
                        f"| mse={mse.item():.6f} | lpips={lp.item():.4f} "
                        f"| psnr={psnr.item():.2f} | keep={keep_prob.mean().item():.3f} "
                        f"| {step_time*1000:.1f}ms/step"
                    )

                # ---- viz ----
                if is_rank0() and viz_every > 0 and (step % viz_every == 0):
                    # Free up VRAM before visualization
                    torch.cuda.empty_cache()
                    
                    with torch.no_grad():
                        patches = temporal_patchify(x, P)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            pred, mae_mask, keep_prob = model(patches)

                        log_tokenizer_viz_wandb(
                            x_btchw=x,
                            pred_btnd=pred,
                            mae_mask_btNp1=mae_mask,
                            patch=P,
                            step=step,
                            max_items=viz_max_items,
                            max_T=viz_max_T,
                        )
                    
                    torch.cuda.empty_cache()
                    
                # ---- ckpt ----
                if is_rank0() and save_every > 0 and (step % save_every == 0):
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    save_ckpt(ckpt_path, step=step, epoch=epoch, model=model, args=args)
                    # also update a "latest" pointer
                    latest = ckpt_dir / "latest.pt"
                    save_ckpt(latest, step=step, epoch=epoch, model=model, args=args)

            start_epoch = epoch + 1

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, default="config/project.yaml")
   
    train(p.parse_args())
