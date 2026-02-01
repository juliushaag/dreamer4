# sharded_frame_dataset.py
import os
import bisect
from pathlib import Path
from typing import Sequence, List, Dict, Union, Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import json
from task_set import TASK_SET

def create_meta_data(outdirs : list[str], output : Path, seq_len : int = 16):

    if isinstance(outdirs, (str, Path)):
        outdirs = [str(outdirs)]
    else:
        outdirs = [str(p) for p in outdirs]

    output = Path(output)

    tasks = list(TASK_SET)
    seq_len = int(seq_len)

    shards: List[Dict] = []
    cum_starts: List[int] = []
    total_starts = 0

    task_dirs = [Path(root) / task for root in outdirs for task in tasks if (Path(root) / task).exists()]
    task_dirs = tqdm(task_dirs, desc="Loading Dataset")

    for task_dir in task_dirs:
        for fname in sorted(os.listdir(task_dir)):
            if not fname.endswith(".pt"):
                continue
            path = task_dir / fname

            try:
                td = torch.load(path, map_location="meta", mmap=True)
            except Exception as e:
                print(f"[ShardedFrameDataset] Skipping shard {path} (load error): {e}")
                continue

            frames = td.get("frames", None)
            if not isinstance(frames, torch.Tensor):
                print(f"[ShardedFrameDataset] Skipping shard {path} (no 'frames' tensor)")
                continue
            if frames.ndim != 4 or frames.shape[1] != 3:
                print(f"[ShardedFrameDataset] Skipping shard {path} (unexpected shape {frames.shape})")
                continue

            N = int(frames.shape[0])

            shards.append(
                {"path": str(path), "num_frames": N}
            )
            cum_starts.append(total_starts)

    
    with open(output, "w") as fp:
        json.dump(dict(
            shards=shards,
            cum_starts=cum_starts,
            total_starts=total_starts
        ),fp=fp)

from miniconf import configclass, config_field

@configclass
class ShardedFrameDataset(Dataset):
    """
    Samples contiguous sequences from preprocessed shards across multiple roots:

      root/<task>/*.pt  with {"frames": (N, 3, H, W) uint8}

    Returns: (T, 3, H, W) float32 in [0,1], where T = seq_len.

    If iid_sampling=True, ignores idx and samples a random starting position
    uniformly over all valid sequence starts across all shards.
    """
    outdirs : list[str] = config_field("processed")
    tasks : list[str] = config_field("tasks")
    seq_len : int = config_field("sequence_length")
    iid_sampling : bool = config_field("iid_sampling")


    def __init__(self):
        super().__init__()

    
        with open( "/mnt/datasets/dreamer4/meta_data.json", "r") as fp:
            data = json.load(fp=fp)

        self.shards: List[Dict] = data["shards"]
        self.cum_starts: List[int] = []
        total_starts = 0
        for shard in self.shards:
            num_starts = shard["num_frames"] - self.seq_len + 1
            shard["num_starts"] = num_starts
            total_starts += num_starts
            self.cum_starts.append(total_starts)
            
        self.total_starts = total_starts
        if self.total_starts == 0:
            print("[ShardedFrameDataset] WARNING: no usable sequences found in outdirs")
        else:
            print(
                f"[ShardedFrameDataset] roots={len(self.outdirs)}, "
                f"shards={len(self.shards):,}, seq_starts={self.total_starts:,}"
            )

        # simple one-shard cache
        self._cache_path: Optional[str] = None
        self._cache_frames: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return self.total_starts

    def _load_shard(self, path: str) -> torch.Tensor:
        if self._cache_path == path and self._cache_frames is not None:
            return self._cache_frames
        td = torch.load(path, map_location="cpu", mmap=True)
        frames = td["frames"]
        self._cache_path = path
        self._cache_frames = frames
        return frames

    def _map_global_start_to_shard(self, global_start: int) -> tuple[int, int]:
        # global_start in [0, total_starts)
        shard_idx = bisect.bisect_right(self.cum_starts, global_start)
        prev_cum = 0 if shard_idx == 0 else self.cum_starts[shard_idx - 1]
        start_idx_in_shard = global_start - prev_cum
        return shard_idx, start_idx_in_shard

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.total_starts == 0:
            raise IndexError("Empty dataset")

        if self.iid_sampling:
            global_start = torch.randint(0, self.total_starts, (1,)).item()
        else:
            if idx < 0 or idx >= self.total_starts:
                raise IndexError(idx)
            global_start = int(idx)

        shard_idx, start = self._map_global_start_to_shard(global_start)

        meta = self.shards[shard_idx]
        frames = self._load_shard(meta["path"])  # (N, 3, H, W)

        end = start + self.seq_len
        seq_u8 = frames[start:end]  # (T, 3, H, W), guaranteed valid by construction
        return seq_u8.to(torch.float32) / 255.0

if __name__ == "__main__":
    
    dirs = [
        "/mnt/datasets/dreamer4/expert-shards",
        "/mnt/datasets/dreamer4/mixed-small-shards",
        "/mnt/datasets/dreamer4/mixed-large-shards",
    ]
    output = "/mnt/datasets/dreamer4/meta_data.json"
    create_meta_data(dirs, output)