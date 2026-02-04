# robocasa_dataset.py
"""
RoboCasa dataset loader for world model training.

Loads demonstrations from RoboCasa HDF5 files containing:
- Images: robot0_agentview_left_image, robot0_agentview_right_image, robot0_eye_in_hand_image
- Actions: (T, 12) - relative end-effector actions
- Rewards: (T,) - sparse rewards
- Proprioception: joint positions, gripper state, etc.

Uses memory-mapped HDF5 access with LRU caching of demo data for efficient
data loading.

Returns a dict per sample with obs, act, act_mask, rew, etc.
"""
import os
import logging
from typing import Dict, List

import h5py
import torch
from torch.utils.data import Dataset
from miniconf import configclass, config_field

from .dataset_utils import DemoCache, TrajectorySubset, collate_batch, split_by_trajectory_generic

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = ['RoboCasaDataset', 'collate_batch', 'TrajectorySubset', 'split_by_trajectory']


@configclass
class RoboCasaDataset(Dataset):
    """RoboCasa dataset for world model training with actions.
    
    Uses HDF5's native chunked access with LRU caching of demo data
    for efficient data loading.
    """
    
    # Config fields - paths
    data_root: str = config_field("data_root")  # e.g., /mnt/datasets/robocasa/v0.1
    tasks: List[str] = config_field("tasks")     # list of task names or ["all"]
    
    # Config fields - dimensions
    seq_len: int = config_field("sequence_length")
    img_size: int = config_field("image_height")
    action_dim: int = 12  # RoboCasa uses 12-dim actions (rel_pos, rel_rot_axis_angle, gripper)
    lang_dim: int = 512
    
    # Config fields - data loading
    cache_mb: int = 2048  # LRU cache size in MB
    hdf5_key: str = "demo_gentex_im128_randcams.hdf5"  # or "demo_im128.hdf5" for multi-stage
    image_key: str = "robot0_agentview_left_image"     # which camera to use
    action_key: str = "actions"                        # "actions" for relative, "actions_abs" for absolute
    
    def __init__(self, verbose: bool = True):
        super().__init__()
        
        self.verbose = verbose
        self.H = self.img_size
        self.W = self.img_size
        self.A = self.action_dim
        self.T = self.seq_len
        
        # LRU cache for demo data
        self._cache = DemoCache(max_bytes=self.cache_mb * 1024 * 1024)
        
        # Keep HDF5 file handles open
        self._hdf5_handles: Dict[str, h5py.File] = {}
        
        # Discover tasks from data_root
        found_tasks = self._discover_tasks()
        
        # Filter to requested tasks
        tasks_filter = set(self.tasks) if self.tasks and self.tasks != ["all"] else None
        if tasks_filter is not None:
            requested = [t for t in self.tasks if t in found_tasks]
            if self.verbose:
                missing = [t for t in tasks_filter if t not in set(found_tasks)]
                print(f"[RoboCasaDataset] Task filter: keeping {len(requested)}/{len(found_tasks)} tasks")
                if missing:
                    print(f"[RoboCasaDataset] WARNING: {len(missing)} requested tasks not found (e.g. {missing[:5]})")
            task_list = requested
        else:
            task_list = list(found_tasks.keys())
        
        # Storage
        self._tasks = []
        self._hdf5_paths = []
        self._demo_keys = []
        self._demo_lengths = []
        self._valid_starts = []  # valid start indices per task: list of (demo_idx, start_idx)
        self._cum_counts = []
        
        # Per-task metadata
        self._lang_embs = []
        self._zero_lang = torch.zeros(self.lang_dim, dtype=torch.float32)
        
        total = 0
        for task in task_list:
            hdf5_path = found_tasks[task]
            
            try:
                f = self._open_hdf5(hdf5_path)
                demo_keys = sorted(
                    [k for k in f['data'].keys() if k.startswith('demo_')],
                    key=lambda x: int(x.split('_')[1])
                )
            except Exception as e:
                if self.verbose:
                    print(f"[RoboCasaDataset] Skipping task={task}: failed to open HDF5: {e}")
                continue
            
            if len(demo_keys) == 0:
                if self.verbose:
                    print(f"[RoboCasaDataset] Skipping task={task}: no demos found")
                continue
            
            # Compute valid starts for each demo
            task_valid_starts = []  # list of (demo_idx, start_idx)
            task_demo_lengths = []
            
            for demo_idx, demo_key in enumerate(demo_keys):
                try:
                    demo = f[f'data/{demo_key}']
                    actions = demo[self.action_key]
                    
                    # Check if image key exists
                    if self.image_key not in demo['obs']:
                        if self.verbose:
                            logger.debug(f"Skipping {task}/{demo_key}: missing image key {self.image_key}")
                        continue
                    
                    demo_len = actions.shape[0]
                    task_demo_lengths.append(demo_len)
                    
                    # Valid starts: must have T frames available
                    if demo_len >= self.T + 1:
                        for start in range(demo_len - self.T):
                            task_valid_starts.append((demo_idx, start))
                
                except Exception as e:
                    logger.debug(f"Skipping {task}/{demo_key}: {e}")
                    continue
            
            if len(task_valid_starts) == 0:
                if self.verbose:
                    print(f"[RoboCasaDataset] Skipping task={task}: no valid windows")
                continue
            
            # Store task info
            self._tasks.append(task)
            self._hdf5_paths.append(hdf5_path)
            self._demo_keys.append(demo_keys)
            self._demo_lengths.append(task_demo_lengths)
            self._valid_starts.append(task_valid_starts)
            
            # Language embedding (can be computed from task name)
            self._lang_embs.append(self._zero_lang)
            
            total += len(task_valid_starts)
            self._cum_counts.append(total)
            
            if self.verbose:
                print(f"[RoboCasaDataset] task={task} demos={len(demo_keys)} valid={len(task_valid_starts)}")

        self.num_tasks = len(self._tasks)
        assert self.num_tasks > 0, f"No tasks found in {self.data_root}"
        if self.verbose:
            print(f"[RoboCasaDataset] Total: {self._cum_counts[-1]:,} sequences across {self.num_tasks} tasks")
    
    def _discover_tasks(self) -> Dict[str, str]:
        """Discover all tasks from the data_root directory.
        
        Returns:
            Dict mapping task_name -> hdf5_path
        """
        found_tasks = {}
        
        # Search for HDF5 files matching the pattern
        for stage in ['single_stage', 'multi_stage']:
            stage_dir = os.path.join(self.data_root, stage)
            if not os.path.exists(stage_dir):
                continue
            
            # Walk through category/task/date structure
            for category in os.listdir(stage_dir):
                category_dir = os.path.join(stage_dir, category)
                if not os.path.isdir(category_dir):
                    continue
                
                for task in os.listdir(category_dir):
                    task_dir = os.path.join(category_dir, task)
                    if not os.path.isdir(task_dir):
                        continue
                    
                    # Find the date folder
                    for date_folder in os.listdir(task_dir):
                        date_dir = os.path.join(task_dir, date_folder)
                        if not os.path.isdir(date_dir):
                            continue
                        
                        # Check for the HDF5 file
                        hdf5_path = os.path.join(date_dir, self.hdf5_key)
                        
                        # Fallback to demo_im128.hdf5 for multi_stage
                        if not os.path.exists(hdf5_path):
                            hdf5_path = os.path.join(date_dir, "demo_im128.hdf5")
                        
                        if os.path.exists(hdf5_path):
                            # Use task name as key
                            task_name = f"{category}/{task}"
                            found_tasks[task_name] = hdf5_path
        
        if self.verbose:
            print(f"[RoboCasaDataset] Found {len(found_tasks)} tasks in {self.data_root}")
        
        return found_tasks
    
    def _open_hdf5(self, path: str) -> h5py.File:
        """Open an HDF5 file with optimized settings for chunked access."""
        if path not in self._hdf5_handles:
            self._hdf5_handles[path] = h5py.File(
                path, 'r',
                libver='latest',
                rdcc_nbytes=16 * 1024 * 1024,  # 16MB chunk cache
                rdcc_nslots=10007,  # prime number for better hashing
            )
        return self._hdf5_handles[path]
    
    def __len__(self):
        return self._cum_counts[-1]
    
    def _lookup(self, idx: int):
        """Convert global index to (task_idx, demo_idx, start_idx)."""
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
        demo_idx, start = self._valid_starts[task_idx][local]
        return task_idx, demo_idx, start
    
    def _load_demo(self, task_idx: int, demo_idx: int) -> Dict[str, torch.Tensor]:
        """Load entire demo data, using cache when available."""
        cache_key = (task_idx, demo_idx)
        
        # Check cache first
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Load from HDF5
        hdf5_path = self._hdf5_paths[task_idx]
        demo_key = self._demo_keys[task_idx][demo_idx]
        
        f = self._open_hdf5(hdf5_path)
        demo = f[f'data/{demo_key}']
        
        # Load entire demo into memory (will be cached)
        # Images: (T, H, W, 3) uint8 -> (T, 3, H, W) uint8
        images_np = demo['obs'][self.image_key][:]
        images = torch.from_numpy(images_np).permute(0, 3, 1, 2).contiguous()
        
        # Resize if needed (keep as uint8 for memory efficiency)
        if images.shape[-2] != self.H or images.shape[-1] != self.W:
            images_f = images.to(torch.float32) / 255.0
            images_f = torch.nn.functional.interpolate(
                images_f,
                size=(self.H, self.W),
                mode='bilinear',
                align_corners=False
            )
            images = (images_f.clamp(0, 1) * 255).to(torch.uint8)
        
        # Actions: (T, action_dim)
        actions = torch.from_numpy(demo[self.action_key][:]).to(torch.float32)
        
        # Rewards: (T,)
        rewards = torch.from_numpy(demo['rewards'][:]).to(torch.float32)
        
        data = {
            'images': images,  # uint8 for memory efficiency
            'actions': actions,
            'rewards': rewards,
        }
        
        # Cache the demo
        self._cache.put(cache_key, data)
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_idx, demo_idx, start = self._lookup(int(idx))
        
        # Load demo data (uses cache)
        data = self._load_demo(task_idx, demo_idx)
        
        # Extract sequence: T+1 observations, T actions/rewards
        obs = data['images'][start:start + self.T + 1].to(torch.float32) / 255.0
        
        # Actions/rewards for transitions obs[t] -> obs[t+1]
        act = data['actions'][start + 1:start + 1 + self.T]
        rew = data['rewards'][start + 1:start + 1 + self.T]
        
        # Pad actions to fixed size if needed
        act_padded = torch.zeros(self.T, self.A, dtype=torch.float32)
        act_dim = min(act.shape[-1], self.A)
        act_padded[:, :act_dim] = torch.nan_to_num(act[:, :act_dim], nan=0.0)
        
        # Action mask (all dimensions valid for RoboCasa)
        act_mask = torch.ones(self.T, self.A, dtype=torch.float32)
        
        return {
            "obs": obs,
            "act": act_padded,
            "act_mask": act_mask,
            "rew": rew,
            "lang_emb": self._lang_embs[task_idx],
            "emb_id": torch.tensor(task_idx, dtype=torch.long),
        }
    
    def close(self):
        """Close all HDF5 file handles and clear cache."""
        for f in self._hdf5_handles.values():
            try:
                f.close()
            except:
                pass
        self._hdf5_handles.clear()
        self._cache.clear()
    
    def __del__(self):
        self.close()


def split_by_trajectory(dataset: RoboCasaDataset, val_fraction: float = 0.1, seed: int = 42) -> tuple:
    """
    Split a RoboCasaDataset into train and validation sets, respecting trajectory boundaries.
    
    For each task, we identify unique demos and split them into train/val sets.
    All sequences from a given demo go entirely to either train or val.
    
    Args:
        dataset: The RoboCasaDataset to split
        val_fraction: Fraction of demos to use for validation (default 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        (train_dataset, val_dataset) tuple of TrajectorySubset objects
    """
    def get_valid_starts(task_idx):
        return dataset._valid_starts[task_idx]
    
    def get_episode_id(item):
        # item is (demo_idx, start_idx), use demo_idx as episode ID
        return item[0]
    
    return split_by_trajectory_generic(
        dataset=dataset,
        num_tasks=dataset.num_tasks,
        get_valid_starts=get_valid_starts,
        get_episode_id=get_episode_id,
        val_fraction=val_fraction,
        seed=seed
    )


if __name__ == "__main__":
    # Test the dataset
    import time
    from pathlib import Path
    from miniconf import MiniConf
    
    # Load config from YAML file (relative to this script)
    script_dir = Path(__file__).parent
    config_path = script_dir / '../../config/robocasa_data.yaml'
    print(f"Loading config from {config_path.resolve()}...")
    conf = MiniConf.load(str(config_path))
    
    print("Creating RoboCasaDataset (with LRU cache)...")
    
    ds = RoboCasaDataset(**conf.select(), verbose=True)
    print(f'\nDataset length: {len(ds)}')
    
    if len(ds) > 0:
        # Test single sample
        print('\n--- Single sample test ---')
        t0 = time.time()
        sample = ds[0]
        t1 = time.time()
        print(f'First sample load time: {(t1-t0)*1000:.2f}ms')
        
        print(f'Sample keys: {list(sample.keys())}')
        print(f'obs shape: {sample["obs"].shape}, dtype: {sample["obs"].dtype}')
        print(f'act shape: {sample["act"].shape}')
        print(f'act_mask shape: {sample["act_mask"].shape}')
        print(f'rew shape: {sample["rew"].shape}')
        print(f'lang_emb shape: {sample["lang_emb"].shape}')
        print(f'emb_id: {sample["emb_id"]}')
        
        # Test train/val split
        print('\n--- Train/val split test ---')
        train_ds, val_ds = split_by_trajectory(ds, val_fraction=0.1)
        print(f'Train size: {len(train_ds)}, Val size: {len(val_ds)}')
        
        # Test loading a batch
        print('\n--- Batch loading test ---')
        t0 = time.time()
        batch = collate_batch([ds[i] for i in range(min(32, len(ds)))])
        t1 = time.time()
        print(f'Batch load time (32 samples): {(t1-t0)*1000:.2f}ms')
        print(f'Batch obs shape: {batch["obs"].shape}')
        print(f'Batch act shape: {batch["act"].shape}')
        
        print('\nDataset working correctly!')
    
    ds.close()
