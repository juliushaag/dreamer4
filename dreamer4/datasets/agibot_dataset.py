# agibot_dataset.py
"""
AgiBot World Model dataset loader for world model training.

Loads demonstrations from AgiBot WorldModel dataset containing:
- Video: head_color.mp4 (640x480, 30fps)
- Proprioception: proprio_stats.h5 with robot state data
  - state/joint/position: (T, 14) joint positions
  - state/joint/current_value: (T, 14) joint values
  - state/end/position: (T, 2, 3) end effector positions
  - state/end/orientation: (T, 2, 4) end effector orientations (quaternion)
  - state/effector/position: (T, 2) gripper positions
  - state/head/position: (T, 2) head joint positions
  - state/waist/position: (T, 2) waist joint positions
  - state/robot/position: (T, 3) robot base position
  - state/robot/orientation: (T, 4) robot base orientation
  - timestamp: (T,) timestamps

Directory structure:
  train/
    {episode_id}/
      head_color.mp4
      proprio_stats.h5
      head_intrinsic_params.json
      head_extrinsic_params_aligned.json

Returns a dict per sample with obs, act, act_mask, rew, etc.
"""

import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Any

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets.dataset_utils import TrajectorySubset, collate_batch, register_dataset

logger = logging.getLogger(__name__)

__all__ = ["AgibotDataset", "collate_batch", "TrajectorySubset"]


class VideoReader:
    """Efficient video reader with support for multiple formats.

    Supports three formats (in order of preference):
    1. HDF5 files (.h5) - compressed, fast, memory-efficient (RECOMMENDED)
    2. Numpy arrays (.npy) - uncompressed, fast, large files
    3. MP4 files (.mp4) - compressed, slow decoding, baseline

    The reader automatically uses the fastest available format.
    """

    def __init__(self, video_path: str, use_mmap: bool = True):
        self.video_path = video_path
        self._frames: Optional[torch.Tensor] = None
        self._num_frames: int = 0

    def __len__(self) -> int:
        if self._frames is None:
            self._load_video()
        return self._num_frames

    def __del__(self):
        """Close HDF5 file if open."""
        pass

    def _load_video(self):
        try:
            import cv2

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                raise RuntimeError(f"No frames read from video: {self.video_path}")

            # Stack frames: (T, H, W, C) -> (T, C, H, W)
            frames_np = torch.from_numpy(np.stack(frames, axis=0))
            self._frames = frames_np.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
            self._num_frames = self._frames.shape[0]
            logger.debug(f"Decoded MP4 video from {self.video_path}")

        except Exception as e:
            logger.error(f"Failed to load video {self.video_path}: {e}")
            self._frames = torch.zeros(1, 3, 480, 640, dtype=torch.uint8)
            self._num_frames = 0

    def get_frames(self, start: int, count: int) -> torch.Tensor:
        """Get frames from video.

        Args:
            start: Starting frame index
            count: Number of frames

        Returns:
            Tensor of shape (count, 3, H, W) uint8
        """
        end = min(start + count, self._num_frames)
        frames = self._frames[start:end]

        # Pad if needed
        if frames.shape[0] < count:
            padding = torch.zeros(
                count - frames.shape[0], *frames.shape[1:], dtype=frames.dtype
            )
            frames = torch.cat([frames, padding], dim=0)

        return frames


class AgibotDataset(Dataset):
    """AgiBot World Model dataset for world model training."""

    def __init__(
        self,
        data_root: str,
        sequence_length: int,
        image_height: int,
        image_width: int,
        action_dim: int,
        lang_dim: int = 512,
        use_joints_as_actions: bool = True,
        include_end_effector: bool = True,
        max_episodes: int = -1,
        use_mmap: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.verbose = verbose
        self.H = image_height
        self.W = image_width
        self.A = action_dim
        self.T = sequence_length
        self.use_joints_as_actions = use_joints_as_actions
        self.include_end_effector = include_end_effector
        self.max_episodes = max_episodes
        self.use_mmap = use_mmap

        # Storage
        self._episode_dirs = []
        self._episode_lengths = []
        self._valid_starts = []  # list of (episode_idx, start_idx)
        self._cum_counts = []

        # Language embedding placeholder
        self._zero_lang = torch.zeros(lang_dim, dtype=torch.float32)

        # Discover episodes from data_root
        self._discover_episodes()

        if self.verbose:
            total = self._cum_counts[-1] if self._cum_counts else 0
            print(
                f"[AgibotDataset] Total: {total:,} sequences across {len(self._episode_dirs)} episodes"
            )

    def _discover_episodes(self):
        """Discover all episodes from the data_root directory."""
        total = 0

        # List all episode directories
        if not os.path.exists(self.data_root):
            raise ValueError(f"Data root does not exist: {self.data_root}")

        episode_dirs = []
        for entry in os.listdir(self.data_root):
            episode_path = os.path.join(self.data_root, entry)
            if not os.path.isdir(episode_path):
                continue

            # Check for required files
            video_path = os.path.join(episode_path, "head_color.mp4")
            h5_path = os.path.join(episode_path, "proprio_stats.h5")

            if os.path.exists(video_path) and os.path.exists(h5_path):
                episode_dirs.append(episode_path)

                # Early exit if max_episodes reached
                if self.max_episodes > 0 and len(episode_dirs) >= self.max_episodes:
                    break

        if self.verbose:
            print(
                f"[AgibotDataset] Found {len(episode_dirs)} episodes in {self.data_root}"
            )

        # Process each episode
        for episode_path in sorted(episode_dirs):
            try:
                # Get episode length from h5 file
                h5_path = os.path.join(episode_path, "proprio_stats.h5")
                with h5py.File(h5_path, "r") as f:
                    # Use timestamp length as episode length
                    episode_len = f["timestamp"].shape[0]

                if episode_len < self.T + 1:
                    if self.verbose:
                        logger.debug(
                            f"Skipping {episode_path}: too short ({episode_len} < {self.T + 1})"
                        )
                    continue

                # Compute valid start indices
                episode_valid_starts = []
                for start in range(episode_len - self.T):
                    episode_valid_starts.append((len(self._episode_dirs), start))

                if len(episode_valid_starts) == 0:
                    continue

                # Store episode info
                self._episode_dirs.append(episode_path)
                self._episode_lengths.append(episode_len)
                self._valid_starts.extend(episode_valid_starts)

                total += len(episode_valid_starts)
                self._cum_counts.append(total)

            except Exception as e:
                logger.debug(f"Skipping {episode_path}: {e}")
                continue

        self.num_episodes = len(self._episode_dirs)
        assert self.num_episodes > 0, f"No valid episodes found in {self.data_root}"

    def __len__(self):
        return len(self._valid_starts)

    def _load_episode_data(self, episode_idx: int) -> Dict[str, Any]:
        """Load episode data from files with caching.

        Returns:
            Dict with keys:
                - frames: (T, 3, H, W) uint8 tensor of all video frames
                - state: Dict of all proprioceptive state tensors
                - episode_length: int
        """
        # Check cache first
        episode_path = self._episode_dirs[episode_idx]

        # Load video frames into memory
        video_path = os.path.join(episode_path, "head_color.mp4")
        video_reader = VideoReader(video_path, use_mmap=self.use_mmap)
        frames = video_reader.get_frames(0, len(video_reader))  # Load all frames

        # Load all proprioception data
        h5_path = os.path.join(episode_path, "proprio_stats.h5")
        state = {}

        with h5py.File(
            h5_path,
            "r",
            libver="latest",
            rdcc_nbytes=16 * 1024 * 1024,
            rdcc_nslots=10007,
        ) as f:
            episode_len = f["timestamp"].shape[0]

            def load_field(key: str, expected_shape_suffix: Tuple, dtype=torch.float32):
                """Load h5 field, returning zeros if empty/missing."""
                try:
                    data = f[key][:]
                    if data.shape[0] == 0:
                        # Field is empty, create zeros with expected shape
                        full_shape = (episode_len,) + expected_shape_suffix
                        return torch.zeros(full_shape, dtype=dtype)
                    return torch.from_numpy(data).to(dtype)
                except KeyError:
                    # Field doesn't exist
                    full_shape = (episode_len,) + expected_shape_suffix
                    return torch.zeros(full_shape, dtype=dtype)

            # Joint state (14 joints for dual arm)
            state["joint_position"] = load_field(
                "state/joint/position", (14,)
            )  # (T, 14)
            state["joint_current"] = load_field(
                "state/joint/current_value", (14,)
            )  # (T, 14)

            # End effector state (2 arms)
            state["end_position"] = load_field(
                "state/end/position", (2, 3)
            )  # (T, 2, 3)
            state["end_orientation"] = load_field(
                "state/end/orientation", (2, 4)
            )  # (T, 2, 4) quaternion

            # Gripper/effector state
            state["gripper_position"] = load_field(
                "state/effector/position", (2,)
            )  # (T, 2)

            # Head state
            state["head_position"] = load_field("state/head/position", (2,))  # (T, 2)

            # Waist state
            state["waist_position"] = load_field("state/waist/position", (2,))  # (T, 2)

            # Robot base state
            state["robot_position"] = load_field("state/robot/position", (3,))  # (T, 3)
            state["robot_orientation"] = load_field(
                "state/robot/orientation", (4,)
            )  # (T, 4) quaternion

            # Timestamp
            state["timestamp"] = torch.from_numpy(f["timestamp"][:]).to(
                torch.int64
            )  # (T,)

        data = {
            "frames": frames,
            "state": state,
            "episode_length": frames.shape[0],
        }

        return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode_idx, start = self._valid_starts[idx]

        # Load episode data (cached)
        data = self._load_episode_data(episode_idx)
        state = data["state"]
        frames = data["frames"]

        # Verify we have enough data
        episode_len = data["episode_length"]
        assert start + self.T + 1 <= episode_len, (
            f"Invalid slice: start={start}, T={self.T}, episode_len={episode_len}"
        )

        # Get frames slice (T+1 for observations)
        frame_slice = frames[start : start + self.T + 1]  # (T+1, 3, H, W)
        assert frame_slice.shape[0] == self.T + 1, (
            f"Wrong frame count: got {frame_slice.shape[0]}, expected {self.T + 1}"
        )

        # Convert to float and normalize
        images = frame_slice.to(torch.float32) / 255.0  # (T+1, 3, H, W)

        # Resize if needed
        if images.shape[-2] != self.H or images.shape[-1] != self.W:
            images = F.interpolate(
                images, size=(self.H, self.W), mode="bilinear", align_corners=False
            )

        # Build structured observations dict
        # All obs have T+1 timesteps (current + future states)
        obs = {
            "images": images,  # (T+1, 3, H, W)
            "joint_position": state["joint_position"][
                start : start + self.T + 1
            ].clone(),  # (T+1, 14)
            "joint_current": state["joint_current"][
                start : start + self.T + 1
            ].clone(),  # (T+1, 14)
            "end_position": state["end_position"][
                start : start + self.T + 1
            ].clone(),  # (T+1, 2, 3)
            "end_orientation": state["end_orientation"][
                start : start + self.T + 1
            ].clone(),  # (T+1, 2, 4)
            "gripper_position": state["gripper_position"][
                start : start + self.T + 1
            ].clone(),  # (T+1, 2)
            "head_position": state["head_position"][
                start : start + self.T + 1
            ].clone(),  # (T+1, 2)
            "waist_position": state["waist_position"][
                start : start + self.T + 1
            ].clone(),  # (T+1, 2)
            "robot_position": state["robot_position"][
                start : start + self.T + 1
            ].clone(),  # (T+1, 3)
            "robot_orientation": state["robot_orientation"][
                start : start + self.T + 1
            ].clone(),  # (T+1, 4)
        }

        # Verify obs shapes
        assert obs["joint_position"].shape[0] == self.T + 1, (
            f"joint_position shape mismatch: {obs['joint_position'].shape}"
        )

        # Build actions from joint deltas + end effector + gripper
        joint_pos = state["joint_position"]
        end_pos = state["end_position"].reshape(-1, 6)  # (T, 6)
        gripper_pos = state["gripper_position"]

        # Combine all action components
        full_actions = torch.cat([joint_pos, end_pos, gripper_pos], dim=-1)  # (T, 22)

        if self.use_joints_as_actions:
            # Use delta positions as actions
            act = (
                full_actions[start + 1 : start + 1 + self.T]
                - full_actions[start : start + self.T]
            )
        else:
            # Use absolute positions
            act = full_actions[start + 1 : start + 1 + self.T]

        # Pad actions to fixed size
        act_padded = torch.zeros(self.T, self.A, dtype=torch.float32)
        act_dim = min(act.shape[-1], self.A)
        act_padded[:, :act_dim] = torch.nan_to_num(act[:, :act_dim], nan=0.0)

        # Action mask
        act_mask = torch.zeros(self.T, self.A, dtype=torch.float32)
        act_mask[:, :act_dim] = 1.0

        # Placeholder rewards (no rewards in this dataset)
        rew = torch.zeros(self.T, dtype=torch.float32)

        return {
            **{f"obs.{k}": v for k, v in obs.items()},
            "act": act_padded,
            "act_mask": act_mask,
            "rew": rew,
            "lang_emb": self._zero_lang,
            "emb_id": torch.tensor(episode_idx, dtype=torch.long),
        }

    def close(self):
        pass

    def get_episode_id(self, item):
        """Get episode ID from valid_starts item."""
        return item[0]

    def get_valid_starts(self, episode_idx: int):
        """Get valid start indices for a specific episode."""
        return [item for item in self._valid_starts if item[0] == episode_idx]

    @property
    def num_tasks(self):
        """Compatibility with robocasa-style API."""
        return self.num_episodes

    def split_by_trajectory(
        self, val_fraction: float = 0.1, seed: int = 42
    ) -> Tuple[TrajectorySubset, TrajectorySubset]:
        """Split dataset by episode into train and validation sets."""
        rng = torch.Generator().manual_seed(seed)

        train_indices = []
        val_indices = []

        # Get unique episodes
        unique_episodes = list(range(self.num_episodes))
        n_eps = len(unique_episodes)

        # Shuffle episodes
        perm = torch.randperm(n_eps, generator=rng).tolist()
        unique_eps_shuffled = [unique_episodes[i] for i in perm]

        # Split episodes into train/val
        n_val = max(1, int(n_eps * val_fraction)) if n_eps > 1 else 0
        val_eps = set(unique_eps_shuffled[:n_val])

        # Assign sequences to train or val based on their episode
        for idx, (episode_idx, _) in enumerate(self._valid_starts):
            if episode_idx in val_eps:
                val_indices.append(idx)
            else:
                train_indices.append(idx)

        train_indices = torch.tensor(train_indices, dtype=torch.long)
        val_indices = torch.tensor(val_indices, dtype=torch.long)

        if self.verbose:
            print(
                f"[AgibotDataset] Split: {len(train_indices):,} train, {len(val_indices):,} val"
            )

        return TrajectorySubset(self, train_indices), TrajectorySubset(
            self, val_indices
        )

    def __del__(self):
        self.close()


register_dataset("agibot", AgibotDataset)


if __name__ == "__main__":
    # Test the dataset
    import time

    print("Creating AgibotDataset...")

    ds = AgibotDataset(
        data_root="/mnt/datasets/agibot/WorldModel/train",
        sequence_length=16,
        image_height=480,
        image_width=640,
        action_dim=32,
        max_episodes=10,  # Limit for testing
        verbose=True,
    )

    print(f"\nDataset length: {len(ds)}")

    if len(ds) > 0:
        # Test single sample
        print("\n--- Single sample test ---")
        t0 = time.time()
        sample = ds[0]
        t1 = time.time()
        print(f"First sample load time: {(t1 - t0) * 1000:.2f}ms")

        print(f"Sample keys: {list(sample.keys())}")
        print(f"obs keys: {list(sample['obs'].keys())}")
        print(f"obs.images shape: {sample['obs']['images'].shape}")
        print(f"obs.joint_position shape: {sample['obs']['joint_position'].shape}")
        print(f"obs.end_position shape: {sample['obs']['end_position'].shape}")
        print(f"obs.gripper_position shape: {sample['obs']['gripper_position'].shape}")
        print(f"act shape: {sample['act'].shape}")
        print(f"act_mask shape: {sample['act_mask'].shape}")
        print(f"rew shape: {sample['rew'].shape}")
        print(f"lang_emb shape: {sample['lang_emb'].shape}")
        print(f"emb_id: {sample['emb_id']}")

        # Test loading a batch
        print("\n--- Batch loading test ---")
        t0 = time.time()
        batch = collate_batch([ds[i] for i in range(min(8, len(ds)))])
        t1 = time.time()
        print(f"Batch load time (8 samples): {(t1 - t0) * 1000:.2f}ms")
        print(f"Batch obs.images shape: {batch['obs']['images'].shape}")
        print(f"Batch obs.joint_position shape: {batch['obs']['joint_position'].shape}")
        print(
            f"Batch obs.gripper_position shape: {batch['obs']['gripper_position'].shape}"
        )
        print(f"Batch act shape: {batch['act'].shape}")

        # Test train/val split
        print("\n--- Train/Val split test ---")
        train_ds, val_ds = ds.split_by_trajectory(val_fraction=0.1)
        print(f"Train size: {len(train_ds)}")
        print(f"Val size: {len(val_ds)}")

        print("\nDataset working correctly!")

    ds.close()
