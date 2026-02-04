# datasets/__init__.py
"""
World model dataset loaders.

Available datasets:
- DMCDataset: DeepMind Control Suite dataset (pre-processed frame shards)
- RoboCasaDataset: RoboCasa kitchen manipulation dataset (HDF5 demos)

Common utilities:
- TrajectorySubset: Dataset wrapper for train/val splits
- DemoCache: LRU cache for demo data
- collate_batch: Batch collation function
"""

from .dataset_utils import (
    DemoCache,
    TrajectorySubset,
    collate_batch,
    split_by_trajectory_generic,
)
from .wm_dataset import (
    DMCDataset,
    split_by_trajectory as dmc_split_by_trajectory,
)
from .robocasa_dataset import (
    RoboCasaDataset,
    split_by_trajectory as robocasa_split_by_trajectory,
)

__all__ = [
    # Datasets
    "DMCDataset",
    "RoboCasaDataset",
    # Utilities
    "DemoCache",
    "TrajectorySubset",
    "collate_batch",
    "split_by_trajectory_generic",
    # Split functions
    "dmc_split_by_trajectory",
    "robocasa_split_by_trajectory",
]
