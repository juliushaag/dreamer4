# datasets/__init__.py
"""
World model dataset loaders.

Available datasets:
- DMCDataset: DeepMind Control Suite dataset (pre-processed frame shards)
- RoboCasaDataset: RoboCasa kitchen manipulation dataset (HDF5 demos)
- AgibotDataset: AgiBot World Model dataset (MP4 videos + H5 proprioception)

Common utilities:
- TrajectorySubset: Dataset wrapper for train/val splits
- DemoCache: LRU cache for demo data
- collate_batch: Batch collation function
"""

from .dataset_utils import (
    TrajectorySubset,
    collate_batch,
)

from .agibot_dataset import AgibotDataset


__all__ = [
    # Datasets
    "DMCDataset",
    "RoboCasaDataset",
    "AgibotDataset",
    # Utilities
    "DemoCache",
    "TrajectorySubset",
    "collate_batch",
    "split_by_trajectory_generic",
    # Split functions
    "dmc_split_by_trajectory",
    "robocasa_split_by_trajectory",
]
