# AGENTS.md - Development Guide for AI Agents

This document provides guidelines for AI agents working on the Dreamer 4 codebase.

## Project Overview

Dreamer 4 is a PyTorch implementation of the Dreamer 4 world model from the paper "Training Agents Inside of Scalable World Models". It consists of:
- **Tokenizer**: VAE-like encoder/decoder for compressing images into latent representations
- **Dynamics Model**: Predicts future latent states given actions
- Both use a block-causal transformer architecture

## Environment

- **Conda Environment**: `dreamer4`
- **Python Version**: 3.10+
- **PyTorch**: With CUDA support
- **CUDA**: Required for training
- **Hardware** RTX 3090, 64GB Ram

### Activate Environment
```bash
conda activate dreamer4
# Or use:
conda run -n dreamer4 <command>
```

## Build/Lint/Test Commands

### Running the Code

```bash
# Single GPU training
python train_tokenizer.py
python train_dynamics.py

# Multi-GPU training (8 GPUs)
torchrun --nproc_per_node=8 train_tokenizer.py
torchrun --nproc_per_node=8 train_dynamics.py

# Interactive web interface
python interactive.py
```

### Testing Individual Components

```bash
# Test model imports
cd /shared/dreamerv4/dreamer4
conda run -n dreamer4 python -c "from dreamer4.model import Tokenizer, Dynamics"

# Test attention module
conda run -n dreamer4 python -c "
import torch
from dreamer4.attention import create_attention, AttentionType
attn = create_attention(AttentionType.GQA, d_model=256, n_heads=8, n_kv_heads=2)
x = torch.randn(2, 16, 256)
y = attn(x)
print(f'Output shape: {y.shape}')
"

# Test MiniConf config system
conda run -n dreamer4 python -c "
from dreamer4.miniconf import MiniConf
config = MiniConf({'num_layers': 2, 'num_heads': 4})
print(config)
"
```

## Code Style Guidelines

### Imports

- **Standard library first**: `math`, `dataclasses`, `enum`, `pathlib`, `typing`
- **Third-party second**: `torch`, `torch.nn`, `torch.nn.functional`, `einops`
- **Local third**: `from dreamer4.miniconf import ...`, `from dreamer4.attention import ...`

Example:
```python
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from einops import rearrange, repeat
from dreamer4.miniconf import configclass, MiniConf, config_field
import lpips
```

### Configuration System

The project uses **MiniConf** for configuration. Use `@configclass` decorator and `config_field()`:

```python
from dreamer4.miniconf import configclass, config_field

@configclass
class Encoder(nn.Module):
    depth: int = config_field("num_layers")
    n_heads: int = config_field("num_heads")
    d_model: int = config_field("latent_dim")
    
    def __init__(self, n_patches: int, d_patch: int):
        conf = MiniConf({'num_layers': 2, 'num_heads': 4, ...})
        encoder = Encoder(n_patches=64, d_patch=192, conf=conf)
```

### Naming Conventions

- **Classes**: `CamelCase` (e.g., `BlockCausalTransformer`, `GroupedQueryAttention`)
- **Functions/variables**: `snake_case` (e.g., `add_sinusoidal_positions`, `d_model`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `AttentionType.MHA`)
- **Private methods**: `_leading_underscore` (e.g., `_repeat_kv`)

### Tensor Shapes

Document tensor shapes in docstrings using standard conventions:
- `B` = batch size, `T` = time steps, `S` = spatial dimension
- `D` = model dimension, `H` = number of heads, `L` = sequence length

Example:
```python
def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
    """x: (B, T, S, D) input tensor"""
```

### Using Einops

Prefer `einops.rearrange` and `einops.repeat` over manual `reshape`/`view`:

```python
# Good
x = rearrange(x_btSd, 'b t s d -> (b t) s d')
y = repeat(self.embed, 'd -> b t d', b=B, t=T)
```

### Attention Module Architecture

The attention system supports both MHA and GQA:
- **MHA**: Standard Multi-Head Attention
- **GQA**: Grouped Query Attention with built-in 2D RoPE and QK normalization

All attention is configured via YAML config files:
```yaml
# tokenizer.yaml or dynamics.yaml
attention_type: "gqa"  # or "mha"
n_kv_heads: 2          # for GQA
rope_base: 10000.0
rope_max_t: 1024
rope_max_s: 1024
```

### Error Handling

- Use assertions for internal invariants:
```python
assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
```

- Use proper exceptions for user-facing errors:
```python
raise ValueError(f"Unknown attention type: {attention_type}")
```

### File Organization

```
dreamer4/
├── model.py           # Main models (Tokenizer, Dynamics, Encoder, Decoder)
├── attention.py       # Attention modules (MHA, GQA, RoPE)
├── miniconf.py       # Configuration system
├── datasets/         # Data loading
├── train_tokenizer.py # Tokenizer training script
├── train_dynamics.py  # Dynamics training script
└── interactive.py     # Web interface
```

### Key Patterns

1. **Gradient Checkpointing**:
```python
if self.gradient_checkpointing and self.training:
    for layer in self.layers:
        x = grad_checkpoint(layer, x, use_reentrant=False)
```

2. **Mixed Precision**:
```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x)
```

3. **No Defaults**:
Everything should be configured from the conf files

## Common Tasks

### Adding New Configuration Options

1. Add to YAML config files (`config/tokenizer.yaml`, `config/dynamics.yaml`)
2. Add as `config_field` in the appropriate `@configclass`
3. Pass through constructor chain