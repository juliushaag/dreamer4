# dataset_utils.py
"""
Shared utilities for world model dataset loaders.

Contains common components used by both DMCDataset and RoboCasaDataset:
- TrajectorySubset: Dataset wrapper for train/val splits
- collate_batch: Batch collation function
- DemoCache: LRU cache for demo data
"""

from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import Dict, Optional, Tuple, List, Union, Any

import torch
from torch.utils.data import Dataset


_DATASETS = dict()


class TrajectoryDataset(ABC, Dataset):
    @abstractmethod
    def split_by_trajectory(self, val_fraction : float, seed : int): ...


def load_datasets(config: dict) -> Tuple[Dataset, Dataset]:
    ds : TrajectoryDataset = _DATASETS[config["type"]](**config)
    return ds.split_by_trajectory(val_fraction=config["val_fraction"], seed=42)


def register_dataset(name: str, cls: type):
    _DATASETS[name] = cls

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


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of samples into a single dict of stacked tensors.

    Supports nested dicts (e.g., obs.images, obs.gripper_position).

    Args:
        batch: List of dicts, each containing tensors or nested dicts

    Returns:
        Dict with same structure, where each leaf tensor is stacked with
        batch dimension first.
    """
    out = {}
    for k in batch[0].keys():
        v = batch[0][k]
        if isinstance(v, dict):
            # Recursively collate nested dicts
            out[k] = collate_batch([b[k] for b in batch])
        elif isinstance(v, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            # For non-tensor values, just collect them in a list
            out[k] = [b[k] for b in batch]
    return out
