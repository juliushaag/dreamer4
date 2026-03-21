# Random Temporal Frame Sampling with Time Delta Encoding

## Overview

Modify the tokenizer to sample random frames from trajectories and use cumulative time deltas as positional encodings. This enables:

1. **Temporal generalization**: Model learns from various time gaps between frames
2. **Dataset agnostic pretraining**: Works across different video datasets and frame rates
3. **Better latent representations**: Tokenizer captures temporal dynamics more effectively

## Configuration Changes

### File: `config/tokenizer.yaml`

Add new config fields:

```yaml
# Sampling configuration
num_sampled_frames: 8      # Number of frames to sample per trajectory
min_trajectory_length: 16   # Skip trajectories shorter than this

# MAE masking (make configurable)
enable_mae_masking: true   # Can be disabled
```

---

## Implementation

### Phase 1: Dataset Pipeline

### File: `datasets/robocasa_dataset.py`, `datasets/wm_dataset.py`

Add sampling logic to return random frame indices:

```python
def sample_frame_indices(self, trajectory_length, num_sampled):
    """
    Sample frame indices uniformly from trajectory.
    
    Args:
        trajectory_length: Total frames in trajectory
        num_sampled: Number of frames to sample
        
    Returns:
        Sorted list of indices, e.g., [0, 5, 12, 20, 35, 48, 67, 95]
    """
    if trajectory_length < num_sampled:
        return None  # Skip this sample
    
    indices = sorted(random.sample(range(trajectory_length), num_sampled))
    return indices
```

Modify `__getitem__`:
- Check trajectory length >= `min_trajectory_length`
- Sample frame indices
- Return frames at sampled indices
- Return indices for position encoding

### Phase 2: Position Encoding with Time Deltas

### File: `attention.py`

Modify `RoPE2D` to accept custom positions:

```python
def forward(self, q, k, t_pos: torch.Tensor, s_pos: torch.Tensor):
    """
    Args:
        q, k: Query and key tensors (N, H, L, D)
        t_pos: Temporal positions - cumulative frame indices (L,)
        s_pos: Spatial positions within each frame (L,)
    """
    # t_pos contains cumulative frame indices from sampled frames
    # e.g., [0, 5, 12, 20, 35, 48, 67, 95]
    # RoPE naturally handles these as sequential positions
    
    def apply_rope(x, positions):
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(N, -1)
        # ... rest of RoPE implementation
```

### Phase 3: Tokenizer Encoder

### File: `model.py`

Add config fields to `Tokenizer`:

```python
@configclass
class Tokenizer(nn.Module):
    # ... existing fields
    
    num_sampled_frames: int = config_field("num_sampled_frames")
    min_trajectory_length: int = config_field("min_trajectory_length")
    enable_mae_masking: bool = config_field("enable_mae_masking")
```

Modify `Encoder.forward`:

```python
def forward(self, patch_tokens_btnd: torch.Tensor, frame_indices: torch.Tensor = None):
    """
    Args:
        patch_tokens: (B, T_sampled, Np, Dp) - sampled frame patches
        frame_indices: (T_sampled,) - cumulative frame indices for positions
    """
    B, T_sampled, Np, Dp = patch_tokens_btnd.shape
    
    # Compute positions from frame indices
    # For cumulative deltas: positions[0] = 0, positions[i] = indices[i] - indices[0]
    positions = frame_indices - frame_indices[0]  # Cumulative from anchor
    
    proj = self.patch_proj(patch_tokens_btnd)
    
    # MAE masking (optional)
    if self.enable_mae_masking:
        proj_masked, mae_mask, keep_prob = self.mae(proj)
    else:
        proj_masked = proj
        mae_mask = torch.zeros(B, T_sampled, Np, 1, dtype=torch.bool, device=proj.device)
        keep_prob = torch.ones(B, T_sampled, 1, device=proj.device, dtype=proj.dtype)
    
    # ... rest of encoder
    # Pass positions to transformer for RoPE
```

### Phase 4: BlockCausalTransformer

### File: `model.py`

Modify `BlockCausalTransformer.forward` to accept positions:

```python
def forward(self, x_btSd: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
    """
    Args:
        x_btSd: (B, T, S, D) - token embeddings
        positions: (T,) - cumulative frame indices for temporal attention
    """
    # Pass positions through to attention layers
    for layer in self.layers:
        x = layer(x, positions=positions)
    return x
```

Modify `SpaceSelfAttentionModality` and `TimeSelfAttention` to use custom positions in RoPE.

### Phase 5: Training Loop

### File: `train_tokenizer.py`

```python
def train_step(self, x: torch.Tensor, frame_indices: torch.Tensor = None, ...):
    """
    Args:
        x: Input frames (B, T_full, C, H, W)
        frame_indices: Sampled indices for current batch (T_sampled,)
    """
    # Sample frames if not provided
    if frame_indices is None:
        frame_indices = self.sample_frame_indices(T_full, self.num_sampled_frames)
    
    # Select sampled frames
    x_sampled = x[:, frame_indices]  # (B, T_sampled, C, H, W)
    
    # Patchify sampled frames
    patches = temporal_patchify(x_sampled, self.P)
    
    # Forward with positions
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred, mae_mask, keep_prob = self(patches, frame_indices=frame_indices)
        # ... rest of training
```

### Inference Mode

```python
@torch.no_grad()
def encode(self, x: torch.Tensor, sample_frames: bool = False):
    """
    Encode frames to latents.
    
    Args:
        x: Input frames (B, T, C, H, W)
        sample_frames: If True, randomly sample frames. If False, use consecutive.
    """
    if sample_frames:
        # Random sampling during eval
        frame_indices = self.sample_frame_indices(x.shape[1], self.num_sampled_frames)
    else:
        # Consecutive frames for inference (default)
        frame_indices = torch.arange(min(self.num_sampled_frames, x.shape[1]))
    
    x_sampled = x[:, frame_indices]
    patches = temporal_patchify(x_sampled, self.P)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        z, _ = self.encoder(patches, frame_indices=frame_indices)
    return z
```

---

## Position Encoding Details

### Frame Indices vs Cumulative Deltas

Given sampled frame indices: `[0, 5, 12, 20, 35, 48, 67, 95]`

**Option A: Raw indices as positions**
```
positions = [0, 5, 12, 20, 35, 48, 67, 95]
```

**Option B: Cumulative from first (RECOMMENDED)**
```
positions = [0, 5-0, 12-0, 20-0, 35-0, 48-0, 67-0, 95-0]
           = [0, 5, 12, 20, 35, 48, 67, 95]
```

In this case they're the same because frame 0 is always included as anchor.

**When first frame != 0:**
```
sampled: [10, 25, 40, 60]
positions (cumulative from anchor): [0, 15, 30, 50]
```

This captures the actual time deltas between frames.

### How RoPE Handles This

RoPE applies sinusoidal transformations based on position indices:
- Position 0: base rotation
- Position 5: 5x rotation  
- Position 12: 12x rotation

The model learns to interpret these positions as temporal distance, enabling generalization across different frame rates and sampling patterns.

---

## Design Decisions

### 1. Why Uniform Sampling?
- Simple and effective
- Ensures coverage across entire trajectory
- No bias toward specific time ranges

### 2. Why Cumulative Deltas?
- Captures actual temporal distance
- Dataset-agnostic (works with different frame rates)
- Model learns "5 frames later" vs "12 frames later"

### 3. Why Always Include Anchor (frame 0)?
- Provides a stable reference point
- Decoder always has something to reconstruct from
- Simpler attention patterns

### 4. MAE Masking
- Keep as-is for spatial patches within each frame
- Can be disabled for experiments

---

## Backward Compatibility

To maintain backward compatibility:

```python
# In config, allow None to use old behavior
num_sampled_frames: Optional[int] = config_field("num_sampled_frames")

# In code:
if self.num_sampled_frames is None:
    # Use all frames consecutively (old behavior)
    frame_indices = torch.arange(T)
else:
    # Use sampling (new behavior)
```

---

## Testing Checklist

- [ ] Sample different frame counts (4, 8, 16)
- [ ] Works with different trajectory lengths
- [ ] Different datasets (RoboCasa, DMC)
- [ ] Different frame resolutions
- [ ] MAE masking disabled works
- [ ] Inference with consecutive frames
- [ ] Training loss converges
- [ ] Cross-dataset generalization

---

## Notes

- **Pretraining**: Works out of the box with any video dataset
- **Position encoding**: Using cumulative indices ensures same relative positions regardless of dataset frame rate
- **Complexity**: Minimal - mostly dataset and config changes
