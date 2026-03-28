

from dataclasses import dataclass
import math
import os
import random
from typing import Optional

import numpy as np
import torch

import torch.distributed as dist

from miniconf import MiniConf

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

@dataclass
class TrainingState:

    model : torch.nn.Module

    opt : Optional[torch.optim.Optimizer] = None
    scheduler : Optional[torch.optim.lr_scheduler.LRScheduler] = None

    conf : Optional[MiniConf] = None

    steps : int = 0
    wandb_run : Optional[str] = None
    best_val : float = float("-inf")

    def load(self, path : str):

        ckpt = torch.load(path, map_location="cpu", weights_only=True)        
        self.conf = MiniConf(ckpt["conf"])  

        self.model.load_state_dict(ckpt["model"])
        if self.opt is not None:
            self.opt.load_state_dict(ckpt["opt"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["sched"])

        self.steps = ckpt["steps"]
        self.wandb_run = ckpt["wandb_run"]
        self.best_val = ckpt["best_val"]

    def save(self, path : str):

        ckpt = dict(
            conf=self.conf.asdict() if self.conf is not None else None,
            opt=self.opt.state_dict(),
            model=self.model.state_dict(),
            scheduler=self.scheduler.state_dict(),
            steps=self.steps,
            wandb_run=self.wandb_run,
            best_val=self.best_val
        )

        torch.save(ckpt, path)        

def create_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a learning rate scheduler with linear warmup followed by cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps (linear increase from 0 to base_lr)
        max_steps: Total training steps
        min_lr: Minimum LR at end of cosine decay
        base_lr: Base learning rate (peak after warmup)
        
    Returns:
        LambdaLR scheduler
    """
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup: 0 -> 1
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay: 1 -> min_lr_ratio
            progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            progress = min(1.0, progress)  # Clamp to avoid going below min_lr
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def num_model_params(model : torch.nn.Module):
    return {
        "trainable" : sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad),
        "frozen" : sum(p.numel() * p.element_size() for p in model.parameters() if not p.requires_grad),
        "all" : sum(p.numel() * p.element_size() for p in model.parameters())
    } 