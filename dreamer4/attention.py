# attention.py
import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from miniconf import config_field, configclass

from torch.nn.attention import sdpa_kernel, SDPBackend

# =============================================================================
# 2D RoPE (Rotary Position Embedding)
# =============================================================================
class RoPE2D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        # inv_freq length = D//4; we'll build full D//2 cos/sin via repeat
        quarter = head_dim // 4
        inv_freq = 1.0 / (base ** (torch.arange(0, quarter).float() / quarter))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k, t_pos, s_pos):
        N, H, L, D = q.shape
        half = D // 2

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)

        def apply_rope_to_half(x, positions):
            if positions.dim() == 1:
                positions = positions.unsqueeze(0).expand(N, -1)
            # angle: (N, L, D//4)
            angle = torch.einsum("nl,d->nld", positions.float(), self.inv_freq)
            # Tile to (N, L, D//2) so it matches x's last dim
            angle = angle.repeat(1, 1, 2)
            cos = angle.cos().unsqueeze(1)  # (N, 1, L, D//2)
            sin = angle.sin().unsqueeze(1)  # (N, 1, L, D//2)
            # x: (N, H, L, D//2) — now shapes are compatible
            return x * cos + rotate_half(x) * sin

        q_s, q_t = q[..., :half], q[..., half:]
        k_s, k_t = k[..., :half], k[..., half:]

        q_s = apply_rope_to_half(q_s, s_pos)
        k_s = apply_rope_to_half(k_s, s_pos)
        q_t = apply_rope_to_half(q_t, t_pos)
        k_t = apply_rope_to_half(k_t, t_pos)

        return torch.cat([q_s, q_t], dim=-1), torch.cat([k_s, k_t], dim=-1)


# =============================================================================
# Grouped Query Attention (GQA) with built-in RoPE
# =============================================================================

@configclass
class GroupedQueryAttention(nn.Module):
    n_heads: int = config_field("heads")
    n_kv_heads: int = config_field("kv_heads")
    head_dim : int = config_field("head_dim")
    dropout: float = config_field("dropout")
    rope_base: float = config_field("rope_base")
    rope_max_t: int = config_field("rope_max_t")
    rope_max_s: int = config_field("rope_max_s")
    bias: bool = config_field("bias")

    def __init__(self, d_model : int):
        super().__init__()
        self.n_groups = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=self.bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=self.bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=self.bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d_model, bias=self.bias)
        
        self.rope = RoPE2D(self.head_dim, base=self.rope_base)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def _get_positions(self, L: int, device: torch.device):
        T, S = self.rope_max_t, self.rope_max_s
        # Fix 4: build a proper 2D grid — T rows × S cols, flattened to length T*S
        # then slice to actual sequence length L
        t_idx = torch.arange(T, device=device).repeat_interleave(S)  # 0,0,...,1,1,...
        s_idx = torch.arange(S, device=device).repeat(T)             # 0,1,...,S,0,1,...
        return t_idx[:L], s_idx[:L]

           
    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        B, L, D = x.shape
        device = x.device

        

        q = rearrange(self.q_proj(x), "b l (h d) -> b h l d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b l (g d) -> b g l d", g=self.n_kv_heads)
        v = rearrange(self.v_proj(x), "b l (g d) -> b g l d", g=self.n_kv_heads)

        q, k = self.q_norm(q), self.k_norm(k)
        t_pos, s_pos = self._get_positions(L, device)
        q, k = self.rope(q, k, t_pos, s_pos)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            y = F.scaled_dot_product_attention(
                q.contiguous(), k.contiguous(), v.contiguous(),
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                enable_gqa=True
            )

        return self.o_proj(rearrange(y, "b h l d -> b l (h d)"))
