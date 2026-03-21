# attention.py
import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from miniconf import config_field, configclass

# =============================================================================
# 2D RoPE (Rotary Position Embedding)
# =============================================================================

class RoPE2D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, t_pos: torch.Tensor, s_pos: torch.Tensor):
        N, H, L, D = q.shape
        
        def apply_rope(x: torch.Tensor, positions: torch.Tensor):
            if positions.dim() == 1:
                positions = positions.unsqueeze(0).expand(N, -1)
            angle = torch.einsum('nl,d->nld', positions, self.inv_freq)
            cos, sin = angle.cos(), angle.sin()
            x1, x2 = x[..., :D//2], x[..., D//2:]
            return torch.cat([x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1), 
                             x1 * sin.unsqueeze(1) + x2 * cos.unsqueeze(1)], dim=-1)
        
        q = apply_rope(q, s_pos)
        k = apply_rope(k, s_pos)
        q = apply_rope(q, t_pos)
        k = apply_rope(k, t_pos)
        return q, k


# =============================================================================
# Multihead Self Attention (MHA)
# =============================================================================
@configclass
class MultiheadSelfAttention(nn.Module):
    n_heads: int = config_field("n_heads")
    dropout: int = config_field("dropout")
    bias: int = config_field("bias")

    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % self.n_heads == 0
        self.head_dim = d_model // self.n_heads
        self.dropout_p = self.dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=self.bias)
        self.out = nn.Linear(d_model, d_model, bias=self.bias)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, *, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b l (three h d) -> three b h l d", three=3, h=self.n_heads).unbind(0)
    
        q, k = self.q_norm(q), self.k_norm(k)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0, is_causal=is_causal)
        return self.out(rearrange(y, "b h l d -> b l (h d)"))


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
        self.dropout_p = self.dropout

        self.q_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=self.bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=self.bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=self.bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d_model, bias=self.bias)
        
        self.rope = RoPE2D(self.head_dim, base=self.rope_base)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def _get_positions(self, L: int, device: torch.device):
        T, S = self.rope_max_t, self.rope_max_s
        t_pos = torch.arange(min(T, L), device=device, dtype=torch.long).repeat(min(S, L))
        s_pos = torch.arange(min(S, L), device=device, dtype=torch.long).repeat(T)[:L]
        return t_pos[:L], s_pos[:L]

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

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        
        return self.o_proj(rearrange(y, "b h l d -> b l (h d)"))
