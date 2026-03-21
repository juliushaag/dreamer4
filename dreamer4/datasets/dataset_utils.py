# dataset_utils.py
"""
Shared utilities for world model dataset loaders.

Contains common components used by both DMCDataset and RoboCasaDataset:
- TrajectorySubset: Dataset wrapper for train/val splits
- collate_batch: Batch collation function
- DemoCache: LRU cache for demo data
"""
from collections import OrderedDict
from typing import Dict, Optional, Tuple, List, Union, Any

import torch
from torch.utils.data import Dataset


class TrajectorySubset(Dataset):
    """
    A subset of a dataset that only includes specific indices.
    Used for train/val splits that respect trajectory boundaries.
    
    Works with any dataset that supports integer indexing.
    """
    def __init__(self, dataset: Dataset, indices: torch.Tensor):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        return self.dataset[int(self.indices[idx].item())]


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a batch of samples into a single dict of stacked tensors.
    
    Args:
        batch: List of dicts, each containing tensors with the same keys
        
    Returns:
        Dict with same keys, where each value is a stacked tensor with
        batch dimension first.
    """
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def split_by_trajectory_generic(
    dataset: Dataset,
    num_tasks: int,
    get_valid_starts: callable,
    get_episode_id: callable,
    val_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[TrajectorySubset, TrajectorySubset]:
    """
    Generic trajectory-based split for any dataset.
    
    Splits a dataset into train and validation sets, ensuring all sequences
    from the same episode/demo go to the same split.
    
    Args:
        dataset: The dataset to split
        num_tasks: Number of tasks in the dataset
        get_valid_starts: Function(task_idx) -> iterable of (local_idx, episode_id) pairs
        get_episode_id: Function to extract episode ID from valid_start entry
        val_fraction: Fraction of episodes to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        (train_dataset, val_dataset) tuple of TrajectorySubset objects
    """
    rng = torch.Generator().manual_seed(seed)
    
    train_indices = []
    val_indices = []
    
    global_offset = 0
    
    for task_idx in range(num_tasks):
        valid_starts = get_valid_starts(task_idx)
        
        # Build mapping from episode to local indices
        ep_to_local_indices = {}
        for local_idx, item in enumerate(valid_starts):
            ep_id = get_episode_id(item)
            if ep_id not in ep_to_local_indices:
                ep_to_local_indices[ep_id] = []
            ep_to_local_indices[ep_id].append(local_idx)
        
        # Get unique episodes
        unique_eps = list(ep_to_local_indices.keys())
        n_eps = len(unique_eps)
        
        if n_eps == 0:
            global_offset += len(list(valid_starts)) if hasattr(valid_starts, '__len__') else 0
            continue
        
        # Shuffle episodes
        perm = torch.randperm(n_eps, generator=rng).tolist()
        unique_eps_shuffled = [unique_eps[i] for i in perm]
        
        # Split episodes into train/val
        n_val = max(1, int(n_eps * val_fraction)) if n_eps > 1 else 0
        val_eps = set(unique_eps_shuffled[:n_val])
        
        # Assign sequences to train or val based on their episode
        num_valid = len(list(valid_starts)) if not hasattr(valid_starts, '__len__') else len(valid_starts)
        for local_idx, item in enumerate(valid_starts):
            global_idx = global_offset + local_idx
            ep_id = get_episode_id(item)
            if ep_id in val_eps:
                val_indices.append(global_idx)
            else:
                train_indices.append(global_idx)
        
        global_offset += num_valid
    
    train_indices = torch.tensor(train_indices, dtype=torch.long)
    val_indices = torch.tensor(val_indices, dtype=torch.long)
    
    return TrajectorySubset(dataset, train_indices), TrajectorySubset(dataset, val_indices)
