# wm_dataset.py
"""
Unified world-model dataset for tokenizer and dynamics training.

Loads sharded frames from `frames_dir/<task>/*.pt` files containing {"frames": (N,3,H,W)}.
Loads demo data (actions, rewards, episodes) from `data_dir/<task>.pt`.

Returns a dict per sample with obs, act, act_mask, rew, etc.
"""
import os
import glob
import json
import bisect
import logging
from pathlib import Path
from typing import Dict, Optional, List

import torch
from torch.utils.data import Dataset
from miniconf import configclass, config_field

from .dataset_utils import DemoCache, TrajectorySubset, collate_batch

logger = logging.getLogger(__name__)


class DMCDataset(Dataset):
    """Unified dataset for world model training with actions."""
    def __init__(
        self,
        data_dirs: List[str],
        frames_dirs: List[str],
        tasks: List[str],
        seq_len: int,
        img_size: int,
        strict_tasks: bool,
        *,
        action_dim: int = 16,
        lang_dim: int = 512,
        shard_size: int = 2048,
        cache_mb: int = 2048,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.verbose = verbose
        self.H = img_size
        self.W = img_size
        self.A = action_dim
        self.T = seq_len
        self.cache_bytes = cache_mb * 1024 * 1024
        
        # Normalize and pair data_dirs with frames_dirs
        data_dirs = [str(x) for x in data_dirs]
        frames_dirs = [str(x) for x in frames_dirs]
        
        if len(data_dirs) != len(frames_dirs):
            if len(data_dirs) == 1:
                data_dirs = data_dirs * len(frames_dirs)
            elif len(frames_dirs) == 1:
                frames_dirs = frames_dirs * len(data_dirs)
            else:
                raise ValueError(f"data_dirs and frames_dirs must have same length (or one must be length-1). "
                                 f"Got {len(data_dirs)} and {len(frames_dirs)}")
        
        self.sources = list(zip(data_dirs, frames_dirs))
        
        # Task metadata (action_dim + text_embedding)
        self.task_meta: Optional[dict] = None
        self._zero_lang = torch.zeros(lang_dim, dtype=torch.float32)
        
        # LRU cache for shards
        self._cache = DemoCache(max_bytes=self.cache_bytes)
        
        # Discover available tasks from data_dirs
        found_tasks = []
        seen = set()
        for dd in data_dirs:
            demo_paths = sorted(glob.glob(os.path.join(dd, "*.pt")))
            for p in demo_paths:
                t = os.path.splitext(os.path.basename(p))[0]
                if t not in seen:
                    seen.add(t)
                    found_tasks.append(t)
        
        # Filter to requested tasks
        tasks_filter = set(tasks).intersection(found_tasks)
        if tasks_filter is not None:
            requested = [t for t in tasks if t in found_tasks]
            task_list = requested
        else:
            task_list = found_tasks
        
        # Storage per task
        self._tasks = []
        self.demo_paths = []
        self.shard_lists = []
        self.seg_cum_frames = []
        self.ep = []
        self.act = []
        self.rew = []
        self.valid_starts = []
        self._cum_counts = []
        
        # Precomputed per-task metadata
        self._emb_ids = []
        self._act_dims = []
        self._act_mask_1d = []
        self._lang_embs = []
        
        total = 0
        for task in task_list:
            # Gather segments for this task from each source
            seg_eps = []
            seg_acts = []
            seg_rews = []
            seg_shards = []
            seg_num_frames = []
            seg_demo_paths = []
            ep_offset = 0
            
            for (dd, fd) in self.sources:
                dp = os.path.join(dd, f"{task}.pt")
                shard_glob = os.path.join(fd, task, "*shard*.pt")
                shards = sorted(glob.glob(shard_glob))
                
                if not os.path.exists(dp) or len(shards) == 0:
                    continue
                
                # Load demo tensors
                try:
                    td = torch.load(dp, map_location="cpu", weights_only=False, mmap=True)
                except Exception as e:
                    logger.warning(f"Skipping task={task} source=({dd},{fd}): torch.load failed: {e}")
                    continue
                
                try:
                    ep = td["episode"].to(torch.int64).cpu()
                    act = td["action"].cpu()
                    rew = td["reward"].cpu()
                except Exception as e:
                    logger.warning(f"Skipping task={task} source=({dd},{fd}): missing keys: {e}")
                    continue
                
                if rew.ndim == 2 and rew.shape[-1] == 1:
                    rew = rew.squeeze(-1)
                    
                rew = rew.to(torch.float32)
                
                if act.ndim == 1:
                    act = act.unsqueeze(-1)
                act = act.to(torch.float32)
                
                N = int(rew.shape[0])
                if act.shape[0] != N or ep.shape[0] != N:
                    logger.warning(f"Skipping task={task} source=({dd},{fd}): length mismatch")
                    continue
                
                # Determine available frames
                try:
                    last = torch.load(shards[-1], map_location="cpu", weights_only=False, mmap=True)
                    last_len = int(last["frames"].shape[0])
                except Exception as e:
                    logger.warning(f"Skipping task={task} source=({dd},{fd}): shard load failed: {e}")
                    continue
                
                N_frames_avail = (len(shards) - 1) * self.shard_size + last_len
                N_eff = min(N, N_frames_avail)
                
                if N_eff < (self.T + 1):
                    logger.debug(f"Skipping task={task} source=({dd},{fd}): not enough frames")
                    continue
                
                ep = ep[:N_eff]
                act = act[:N_eff]
                rew = rew[:N_eff]
                
                # Make episode IDs unique across segments
                if ep.numel() > 0:
                    seg_max = int(ep.max().item())
                else:
                    seg_max = 0
                ep = ep + ep_offset
                ep_offset += seg_max + 1
                
                seg_eps.append(ep)
                seg_acts.append(act)
                seg_rews.append(rew)
                seg_shards.append(shards)
                seg_num_frames.append(int(N_eff))
                seg_demo_paths.append(dp)
            
            if len(seg_eps) == 0:
                if self.verbose:
                    print(f"[WMDataset] Skipping task={task}: no valid sources")
                continue
            
            # Concatenate segments
            ep = torch.cat(seg_eps, dim=0)
            act = torch.cat(seg_acts, dim=0)
            rew = torch.cat(seg_rews, dim=0)
            N_eff = int(rew.shape[0])
            
            # Compute valid starts
            start_count = N_eff - self.T
            ep_ok = (ep[:start_count] == ep[self.T:self.T + start_count])
            
            # Filter invalid transitions
            act_nan = torch.isnan(act).any(dim=-1)
            rew_nan = torch.isnan(rew)
            step_ok = ~(act_nan | rew_nan)
            step_ok2 = step_ok[1:]
            
            cs = torch.cumsum(step_ok2.to(torch.int32), dim=0)
            end = torch.arange(start_count) + (self.T - 1)
            prev = torch.arange(start_count) - 1
            prev_cs = torch.zeros(start_count, dtype=cs.dtype)
            m = prev >= 0
            prev_cs[m] = cs[prev[m]]
            window_sum = cs[end] - prev_cs
            window_ok = (window_sum == self.T)
            
            valid = ep_ok & window_ok
            valid_idx = valid.nonzero(as_tuple=False).flatten()
            
            if valid_idx.numel() == 0:
                if self.verbose:
                    print(f"[WMDataset] Skipping task={task}: no valid windows")
                continue
            
            # Per-task action_dim from tasks.json
            act_dim = self.A
            if self.task_meta is not None and task in self.task_meta:
                md = self.task_meta[task]
                if "action_dim" in md:
                    try:
                        act_dim = int(md["action_dim"])
                    except Exception:
                        act_dim = self.A
            act_dim = max(0, min(act_dim, self.A))
            
            act_mask_1d = torch.zeros(self.A, dtype=torch.float32)
            if act_dim > 0:
                act_mask_1d[:act_dim] = 1.0
            
            # Per-task language embedding
            lang = self._zero_lang
            if self.task_meta is not None and task in self.task_meta and "text_embedding" in self.task_meta[task]:
                te = self.task_meta[task]["text_embedding"]
                l = torch.tensor(te, dtype=torch.float32)
                if l.numel() == self.lang_dim:
                    lang = l
            
            # Store
            self._tasks.append(task)
            self.demo_paths.append(seg_demo_paths)
            self.shard_lists.append(seg_shards)
            
            cum = []
            s = 0
            for nf in seg_num_frames:
                s += int(nf)
                cum.append(s)
            self.seg_cum_frames.append(cum)
            
            self.ep.append(ep)
            self.act.append(act)
            self.rew.append(rew)
            self.valid_starts.append(valid_idx)
            
            task_idx = len(self._tasks) - 1
            self._emb_ids.append(torch.tensor(task_idx, dtype=torch.long))
            self._act_dims.append(act_dim)
            self._act_mask_1d.append(act_mask_1d)
            self._lang_embs.append(lang)
            
            total += int(valid_idx.numel())
            self._cum_counts.append(total)
            
            if self.verbose:
                print(f"[WMDataset] task={task} segs={len(seg_shards)} N={N_eff} valid={valid_idx.numel()} act_dim={act_dim}")
        
        self.num_tasks = len(self._tasks)
        assert self.num_tasks > 0, "No tasks found with both demo .pt and frame shards."
        if self.verbose:
            print(f"[WMDataset] Total: {self._cum_counts[-1]:,} sequences across {self.num_tasks} tasks")

    def __len__(self):
        return self._cum_counts[-1]

    def _lookup(self, idx: int):
        lo, hi = 0, len(self._cum_counts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum_counts[mid]:
                hi = mid
            else:
                lo = mid + 1
        task_idx = lo
        prev = 0 if task_idx == 0 else self._cum_counts[task_idx - 1]
        local = idx - prev
        start = int(self.valid_starts[task_idx][local].item())
        return task_idx, start

    def _load_shard_frames(self, task_idx: int, seg_idx: int, shard_idx: int) -> torch.Tensor:
        key = (task_idx, seg_idx, shard_idx)

        path = self.shard_lists[task_idx][seg_idx][shard_idx]
        td = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        frames = td["frames"]

        # Normalize to (N,3,H,W)
        if frames.ndim == 4 and frames.shape[-1] == 3 and frames.shape[1] != 3:
            frames = frames.permute(0, 3, 1, 2).contiguous()

        # Ensure uint8
        if frames.dtype != torch.uint8:
            frames_f = frames.to(torch.float32)
            mx = float(frames_f.max().item()) if frames_f.numel() > 0 else 0.0
            if mx > 1.5:
                frames = frames_f.clamp(0, 255).to(torch.uint8)
            else:
                frames = (frames_f.clamp(0, 1) * 255.0).to(torch.uint8)

        if frames.shape[-2] != self.H or frames.shape[-1] != self.W:
            raise RuntimeError(f"Shard frame size mismatch: {tuple(frames.shape[-2:])} != {(self.H, self.W)}")

        return frames

    def _get_frames(self, task_idx: int, start: int, length: int) -> torch.Tensor:
        out = []
        idx = int(start)
        remaining = int(length)
        seg_cum = self.seg_cum_frames[task_idx]

        while remaining > 0:
            seg_idx = bisect.bisect_right(seg_cum, idx)
            prev_cum = 0 if seg_idx == 0 else seg_cum[seg_idx - 1]
            seg_end = seg_cum[seg_idx]
            local_idx = idx - prev_cum

            shard_idx = local_idx // self.shard_size
            off = local_idx % self.shard_size

            frames = self._load_shard_frames(task_idx, seg_idx, shard_idx)
            take = min(remaining, frames.shape[0] - off, seg_end - idx)

            if take <= 0:
                raise RuntimeError(f"Frame indexing error: task={self._tasks[task_idx]} idx={idx}")

            out.append(frames[off:off + take])
            idx += take
            remaining -= take

        return torch.cat(out, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_idx, start = self._lookup(int(idx))

        obs = self._get_frames(task_idx, start, self.T + 1)  # (T+1,3,H,W) uint8
        obs = obs.to(torch.float32) / 255.0

        # Transition from obs[t] -> obs[t+1] uses action/reward at index (t+1)
        act = self.act[task_idx][start + 1 : start + 1 + self.T]
        rew = self.rew[task_idx][start + 1 : start + 1 + self.T]

        Ad = int(self._act_dims[task_idx])
        act_padded = torch.zeros(self.T, self.A, dtype=torch.float32)
        if Ad > 0:
            act_padded[:, :Ad] = torch.nan_to_num(act[:, :Ad], nan=0.0)

        act_mask = self._act_mask_1d[task_idx][None, :].expand(self.T, self.A).contiguous()

        return {
            "obs": obs,
            "act": act_padded,
            "act_mask": act_mask,
            "rew": rew,
            "lang_emb": self._lang_embs[task_idx],
            "emb_id": self._emb_ids[task_idx],
        }


# Re-export collate_batch for backward compatibility
# (already imported from dataset_utils at top of file)


def split_by_trajectory(dataset: DMCDataset, val_fraction: float = 0.1, seed: int = 42) -> tuple:
    """
    Split a WMDataset into train and validation sets, respecting trajectory boundaries.
    
    For each task, we identify unique episodes and split them into train/val sets.
    All sequences from a given episode go entirely to either train or val.
    
    Args:
        dataset: The WMDataset to split
        val_fraction: Fraction of episodes to use for validation (default 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        (train_dataset, val_dataset) tuple of TrajectorySubset objects
    """
    rng = torch.Generator().manual_seed(seed)
    
    train_indices = []
    val_indices = []
    
    global_offset = 0
    
    for task_idx in range(dataset.num_tasks):
        valid_starts = dataset.valid_starts[task_idx]  # Tensor of valid start indices
        ep_ids = dataset.ep[task_idx]  # Episode IDs for all frames
        
        # For each valid start, get its episode ID
        start_ep_ids = ep_ids[valid_starts]  # Episode ID for each valid sequence
        
        # Get unique episodes
        unique_eps = torch.unique(start_ep_ids)
        n_eps = len(unique_eps)
        
        if n_eps == 0:
            global_offset += len(valid_starts)
            continue
        
        # Shuffle episodes
        perm = torch.randperm(n_eps, generator=rng)
        unique_eps_shuffled = unique_eps[perm]
        
        # Split episodes into train/val
        n_val = max(1, int(n_eps * val_fraction)) if n_eps > 1 else 0
        val_eps = set(unique_eps_shuffled[:n_val].tolist())
        
        # Assign sequences to train or val based on their episode
        for local_idx, ep_id in enumerate(start_ep_ids.tolist()):
            global_idx = global_offset + local_idx
            if ep_id in val_eps:
                val_indices.append(global_idx)
            else:
                train_indices.append(global_idx)
        
        global_offset += len(valid_starts)
    
    train_indices = torch.tensor(train_indices, dtype=torch.long)
    val_indices = torch.tensor(val_indices, dtype=torch.long)
    
    return TrajectorySubset(dataset, train_indices), TrajectorySubset(dataset, val_indices)


if __name__ == "__main__":
    from miniconf import MiniConf
    
    conf = MiniConf.load('../config/project.yaml')
    print('Config loaded')
    
    ds = DMCDataset(**conf.select('data'))
    print(f'Dataset length: {len(ds)}')
    
    if len(ds) > 0:
        sample = ds[0]
        print(f'Sample keys: {list(sample.keys())}')
        print(f'obs shape: {sample["obs"].shape}, dtype: {sample["obs"].dtype}')
        print(f'act shape: {sample["act"].shape}')
        print(f'act_mask shape: {sample["act_mask"].shape}')
        print(f'rew shape: {sample["rew"].shape}')
        print('Dataset working correctly!')
