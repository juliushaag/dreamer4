# model.py
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from einops import rearrange
from miniconf import configclass, MiniConf, config_field
import lpips


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def create_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a learning rate scheduler with linear warmup followed by cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps (linear increase from 0 to base_lr)
        max_steps: Total training steps
        min_lr: Minimum LR at end of cosine decay
        base_lr: Base learning rate (peak after warmup)
        
    Returns:
        LambdaLR scheduler
    """
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup: 0 -> 1
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay: 1 -> min_lr_ratio
            progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            progress = min(1.0, progress)  # Clamp to avoid going below min_lr
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class Modality(IntEnum):
    LATENT = -1
    IMAGE = 0
    ACTION = 1
    PROPRIO = 2
    REGISTER = 3
    SPATIAL = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7


@dataclass(frozen=True)
class TokenLayout:
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> torch.Tensor:
        parts = []
        if self.n_latents > 0:
            parts.append(torch.full((self.n_latents,), int(Modality.LATENT), dtype=torch.int32))
        for m, n in self.segments:
            if n > 0:
                parts.append(torch.full((n,), int(m), dtype=torch.int32))
        return torch.cat(parts, dim=0) if parts else torch.zeros((0,), dtype=torch.int32)

    def slices(self) -> Dict[Modality, slice]:
        idx = 0
        out: Dict[Modality, slice] = {}
        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents)
            idx += self.n_latents
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out
    
def temporal_patchify(videos_btchw: torch.Tensor, patch: int) -> torch.Tensor:
    """
    videos: (B,T,C,H,W) float in [0,1]
    returns: (B,T,Np,Dp) where Dp = patch*patch*C and Np = (H/patch)*(W/patch)
    """
    B, T, C, H, W = videos_btchw.shape
    x = rearrange(videos_btchw, 'b t c h w -> (b t) c h w')
    cols = F.unfold(x, kernel_size=patch, stride=patch)          # (BT, C*pp, Np)
    return rearrange(cols, '(b t) dp np -> b t np dp', b=B, t=T)

def temporal_unpatchify(patches_btnd: torch.Tensor, H: int, W: int, C: int, patch: int) -> torch.Tensor:
    """
    patches: (B,T,Np,Dp) -> (B,T,C,H,W)
    """
    B, T, Np, Dp = patches_btnd.shape
    x = rearrange(patches_btnd, 'b t np dp -> (b t) dp np')
    out = F.fold(x, output_size=(H, W), kernel_size=patch, stride=patch)  # (BT, C, H, W)
    return rearrange(out, '(b t) c h w -> b t c h w', b=B, t=T)

def _sinusoid_table_cached(n: int, d: int, base: float, device_str: str) -> torch.Tensor:
    """Cached sinusoidal position table computation."""
    device = torch.device(device_str)
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)  # (n,1)
    i   = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)  # (1,d)
    k   = torch.floor(i / 2.0)
    div = torch.exp(-(2.0 * k) / max(1.0, float(d)) * math.log(base))
    ang = pos * div
    return torch.where((i % 2) == 0, torch.sin(ang), torch.cos(ang))  # (n,d) fp32


def sinusoid_table(n: int, d: int, base: float = 10000.0, device=None) -> torch.Tensor:
    device_str = str(device) if device is not None else "cpu"
    return _sinusoid_table_cached(n, d, base, device_str)


def add_sinusoidal_positions(tokens_btSd: torch.Tensor) -> torch.Tensor:
    B, T, S, D = tokens_btSd.shape
    device = tokens_btSd.device
    pos_t = sinusoid_table(T, D, device=device)  # fp32
    pos_s = sinusoid_table(S, D, device=device)  # fp32
    pos = (pos_t[None, :, None, :] + pos_s[None, None, :, :]) * (1.0 / math.sqrt(D))
    return tokens_btSd + pos.to(dtype=tokens_btSd.dtype)


class MAEReplacer(nn.Module):
    def __init__(self, d_model: int, p_min: float = 0.0, p_max: float = 0.9):
        super().__init__()
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.mask_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, patches_btnd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        patches: (B,T,Np,D)
        returns:
          replaced: (B,T,Np,D)
          mae_mask: (B,T,Np,1) bool, True where masked (must reconstruct)
          keep_prob:(B,T,1) float
        """
        B, T, Np, D = patches_btnd.shape
        device = patches_btnd.device

        # fast path: deterministic "no MAE"
        if self.p_min == 0.0 and self.p_max == 0.0:
            keep_prob = torch.ones((B, T, 1), device=device, dtype=patches_btnd.dtype)
            mae_mask = torch.zeros((B, T, Np, 1), device=device, dtype=torch.bool)
            return patches_btnd, mae_mask, keep_prob

        p_bt = torch.empty((B, T), device=device).uniform_(self.p_min, self.p_max)
        keep_prob = (1.0 - p_bt).unsqueeze(-1)                          # (B,T,1)
        keep = (torch.rand((B, T, Np), device=device) < keep_prob)      # (B,T,Np)
        keep_ = keep.unsqueeze(-1)
        mask_tok = self.mask_token.to(dtype=patches_btnd.dtype)
        replaced = torch.where(keep_, patches_btnd, mask_tok.view(1, 1, 1, D))
        mae_mask = (~keep_).to(torch.bool)
        return replaced, mae_mask, keep_prob


class RMSNorm(nn.Module):
    """RMSNorm with optional fused implementation (PyTorch 2.4+)."""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.d = d
        self.scale = nn.Parameter(torch.ones(d))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.d,), self.scale, self.eps)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio * (2.0 / 3.0))
        self.fc_in = nn.Linear(d_model, 2 * hidden)
        self.fc_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc_in(x).chunk(2, dim=-1)
        h = u * F.silu(v)
        h = self.drop(h)
        y = self.fc_out(h)
        y = self.drop(y)
        return y


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, qkv_bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x_nld: torch.Tensor, *, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        """
        x: (N,L,D)
        attn_mask: bool, True means "allowed to attend" (for torch SDPA), broadcastable to (N,1,L,L) or (N,H,L,L)
        """
        N, L, D = x_nld.shape
        # Fused reshape: directly to (N, L, 3, H, head_dim) then unbind
        qkv = self.qkv(x_nld).view(N, L, 3, self.n_heads, self.head_dim)
        # Unbind and transpose in one step using permute for better memory access
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N, H, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Views, no copy

        drop = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=is_causal)
        y = y.transpose(1, 2).reshape(N, L, D)  # reshape instead of view after transpose
        return self.out(y)


class SpaceSelfAttentionModality(nn.Module):
    def __init__(self, d_model: int, n_heads: int, modality_ids: torch.Tensor, n_latents: int, mode: str, dropout: float):
        super().__init__()
        self.n_latents = int(n_latents)
        self.mode = mode
        self.register_buffer("modality_ids", modality_ids.to(torch.int32), persistent=False)

        S = int(self.modality_ids.numel())
        allow = self._build_allow(S)                               # (S,S) True=allowed
        attn_mask = allow.unsqueeze(0).unsqueeze(0)                # (1,1,S,S) True=allowed (PyTorch SDPA bool mask)
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        # Note: use self.attn_mask buffer in forward, not a separate attribute
        self._use_mask = self.mode not in ("wm_agent",)

        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def _build_allow(self, S: int) -> torch.Tensor:
        device = self.modality_ids.device
        q_idx = torch.arange(S, device=device).unsqueeze(1)  # (S,1)
        k_idx = torch.arange(S, device=device).unsqueeze(0)  # (1,S)

        is_q_lat = q_idx < self.n_latents
        is_k_lat = k_idx < self.n_latents

        q_mod = self.modality_ids[q_idx]
        k_mod = self.modality_ids[k_idx]
        same_mod = (q_mod == k_mod)

        if self.mode == "encoder":
            allow_lat_q = torch.ones((S, S), dtype=torch.bool, device=device)
            allow_nonlat_q = same_mod
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "decoder":
            allow_lat_q = is_k_lat
            allow_nonlat_q = same_mod | is_k_lat
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "wm_agent":
            # full mixing across modalities
            return torch.ones((S, S), dtype=torch.bool, device=device)

        if self.mode == "wm_agent_isolated":
            # non-agent tokens: can attend to everything EXCEPT agent tokens
            # agent tokens: attend only to agent tokens (keeps them inert in pretrain)
            is_q_agent = (q_mod == int(Modality.AGENT))
            is_k_agent = (k_mod == int(Modality.AGENT))

            allow = torch.ones((S, S), dtype=torch.bool, device=device)

            # non-agent queries cannot see agent keys
            allow_non_agent_q = ~is_q_agent
            allow = torch.where(allow_non_agent_q, ~is_k_agent, allow)

            # agent queries only see agent keys
            allow = torch.where(is_q_agent, is_k_agent, allow)
            return allow

        raise ValueError(f"Unsupported mode for tokenizer/wm: {self.mode}")

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        x = x_btSd.reshape(B * T, S, D)
        # Use registered buffer (moves with model to correct device)
        mask = self.attn_mask if self._use_mask else None
        y = self.attn(x, attn_mask=mask, is_causal=False)
        return y.reshape(B, T, S, D)


class TimeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, latents_only: bool, n_latents: int):
        super().__init__()
        self.latents_only = bool(latents_only)
        self.n_latents = int(n_latents)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

        self.forward = self.forward_latents_only if self.latents_only else self.forward_latents_all

    def forward_latents_all(self, x_btSd: torch.Tensor):
        B, T, S, D = x_btSd.shape
        x_bst = rearrange(x_btSd, 'b t s d -> (b s) t d')
        out = self.attn(x_bst, is_causal=True)
        return rearrange(out, '(b s) t d -> b t s d', b=B, s=S)

    def forward_latents_only(self, x_btSd: torch.Tensor):
        B, T, S, D = x_btSd.shape
        L = self.n_latents
        # Extract latents and reshape for temporal attention
        lat = rearrange(x_btSd[:, :, :L, :], 'b t l d -> (b l) t d')
        out = self.attn(lat, is_causal=True)
        out = rearrange(out, '(b l) t d -> b t l d', b=B, l=L)
        # In-place update for the latent portion (avoids full tensor clone)
        # Create output tensor by concatenating updated latents with unchanged rest
        return torch.cat([out, x_btSd[:, :, L:, :]], dim=2)

class BlockCausalLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        layer_index: int,
        time_every: int,
        latents_only_time: bool,
    ):
        super().__init__()
        self.do_time = ((layer_index + 1) % time_every == 0)

        self.res1 = nn.Sequential(
            RMSNorm(d_model),
            SpaceSelfAttentionModality(d_model, n_heads, modality_ids, n_latents, space_mode, dropout),
            nn.Dropout(dropout, True)
        )

        self.res2 = nn.Identity() if not self.do_time else nn.Sequential(
            RMSNorm(d_model),
            TimeSelfAttention(d_model, n_heads, dropout, latents_only_time, n_latents),
            nn.Dropout(dropout, True),
        )
        
        self.res3 = nn.Sequential(
            RMSNorm(d_model),
            MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout),
            nn.Dropout(dropout, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.res1(x)
        if self.do_time: 
            x = x + self.res2(x)
        x = x + self.res3(x)
        return x


class BlockCausalTransformer(nn.Module):
    """Sequential stack of BlockCausalLayer modules with optional gradient checkpointing."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        time_every: int,
        latents_only_time: bool,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.Sequential(*[
            BlockCausalLayer(
                d_model=d_model,
                n_heads=n_heads,
                n_latents=n_latents,
                modality_ids=modality_ids,
                space_mode=space_mode,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                layer_index=i,
                time_every=time_every,
                latents_only_time=latents_only_time,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            # Checkpoint each layer to trade compute for memory
            for layer in self.layers:
                x = grad_checkpoint(layer, x, use_reentrant=False)
            return x
        return self.layers(x)

@configclass
class Encoder(nn.Module):
    depth : int = config_field("num_layers")
    n_heads : int = config_field("num_heads")
    d_model : int = config_field("latent_dim")
    d_bottleneck : int = config_field("bottleneck_dim")

    n_latents : int = config_field("num_latents")

    mlp_ratio : float = config_field("mlp_ratio")
    dropout : float = config_field("dropout")
    time_every : int = config_field("time_embedding_every")

    latents_only_time : bool = config_field("latents_only_time")
    gradient_checkpointing : bool = config_field("gradient_checkpointing")

    mae_p_min : float = config_field("mae_p_min")
    mae_p_max : float = config_field("mae_p_max")

    def __init__(self, n_patches: int, d_patch: int):
        super().__init__()

        self.n_patches = n_patches  # number of spatial patches
        self.d_patch = d_patch      # dimension of each patch (P*P*C)

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.patch_proj = nn.Linear(self.d_patch, self.d_model)
        self.bottleneck_proj = nn.Linear(self.d_model, self.d_bottleneck)

        self.layout = TokenLayout(n_latents=self.n_latents, segments=((Modality.IMAGE, self.n_patches),))
        modality_ids = self.layout.modality_ids()  # CPU buffer, moves with .to(device)

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model, n_heads=self.n_heads, depth=self.depth,
            n_latents=self.n_latents, modality_ids=modality_ids, space_mode="encoder",
            dropout=self.dropout, mlp_ratio=self.mlp_ratio,
            time_every=self.time_every, latents_only_time=self.latents_only_time,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        self.mae = MAEReplacer(d_model=self.d_model, p_min=self.mae_p_min, p_max=self.mae_p_max)

        self.latents = nn.Parameter(torch.empty(self.n_latents, self.d_model))
        nn.init.normal_(self.latents, std=0.02)

    def forward(self, patch_tokens_btnd: torch.Tensor):

        B, T, Np, Dp = patch_tokens_btnd.shape
        assert Np == self.n_patches
        assert Dp == self.d_patch

        proj = self.patch_proj(patch_tokens_btnd)            # (B,T,Np,D)
        proj_masked, mae_mask, keep_prob = self.mae(proj)    # (B,T,Np,D), (B,T,Np,1), (B,T,1)

        lat = self.latents.view(1, 1, self.n_latents, -1).expand(B, T, -1, -1)
        tokens = torch.cat([lat, proj_masked], dim=2)        # (B,T,S,D)
        tokens = add_sinusoidal_positions(tokens)

        enc = self.transformer(tokens)
        z = torch.tanh(self.bottleneck_proj(enc[:, :, :self.n_latents, :]))
        return z, (mae_mask, keep_prob)


@configclass
class Decoder(nn.Module):

    depth : int = config_field("num_layers")
    n_heads : int = config_field("num_heads")
    d_model : int = config_field("latent_dim")
    d_bottleneck : int = config_field("bottleneck_dim")

    n_latents : int = config_field("num_latents")

    mlp_ratio : float = config_field("mlp_ratio")
    dropout : float = config_field("dropout")
    time_every : int = config_field("time_embedding_every")

    latents_only_time : bool = config_field("latents_only_time")
    gradient_checkpointing : bool = config_field("gradient_checkpointing")

    def __init__(self, n_patches: int, d_patch: int):
        super().__init__()

        self.n_patches = n_patches  # number of spatial patches
        self._d_patch = d_patch     # dimension of each patch (P*P*C)

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


        self.up_proj = nn.Linear(self.d_bottleneck, self.d_model)
        self.patch_queries = nn.Parameter(torch.empty(self.n_patches, self.d_model))
        nn.init.normal_(self.patch_queries, std=0.02)

        self.patch_head = nn.Linear(self.d_model, self._d_patch)

        self.layout = TokenLayout(n_latents=self.n_latents, segments=((Modality.IMAGE, self.n_patches),))
        modality_ids = self.layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model, n_heads=self.n_heads, depth=self.depth,
            n_latents=self.n_latents, modality_ids=modality_ids, space_mode="decoder",
            dropout=self.dropout, mlp_ratio=self.mlp_ratio,
            time_every=self.time_every, latents_only_time=self.latents_only_time,
            gradient_checkpointing=self.gradient_checkpointing,
        )



    def forward(self, z_btLd: torch.Tensor) -> torch.Tensor:
        B, T, L, _ = z_btLd.shape
        assert L == self.n_latents

        lat = torch.tanh(self.up_proj(z_btLd))                                 # (B,T,L,D)
        qry = self.patch_queries.view(1, 1, self.n_patches, -1).expand(B, T, -1, -1)
        tokens = torch.cat([lat, qry], dim=2)                                  # (B,T,S,D)
        tokens = add_sinusoidal_positions(tokens)

        x = self.transformer(tokens)
        x_p = x[:, :, L:, :]
        return torch.sigmoid(self.patch_head(x_p))                             # (B,T,Np,Dp)

@configclass
class Tokenizer(nn.Module):

    H : int = config_field("data/image_height")
    W : int = config_field("data/image_width")
    C : int = config_field("data/image_channels")

    num_patches : int = config_field("num_patches", ge=1)
    num_latents : int = config_field("num_latents", ge=1)

    d_bottleneck : int = config_field("bottleneck_dim")

    lr : float = config_field("optim/lr")
    weight_decay : float = config_field("optim/weight_decay")
    max_steps : int = config_field("optim/max_steps")
    warmup_steps : int = config_field("optim/warmup_steps")
    min_lr : float = config_field("optim/min_lr")

    lpips_fn : str = config_field("lpips/net")
    lpips_frac : float = config_field("lpips/frac")
    lpips_weight : float = config_field("lpips/weight")

    compile : bool = config_field("compile")

    P : int = config_field("num_patches")  # This is actually patch_size (kernel/stride)
        

    def __init__(self, device : str):
        super().__init__()

        
        assert self.H % self.P == 0 and self.W % self.P == 0
        self.n_patches = (self.H // self.P) * (self.W // self.P)  # number of patches
        self.d_patch = self.P * self.P * self.C              # patch dimension (pixels per patch)

        self.encoder = Encoder(n_patches=self.n_patches, d_patch=self.d_patch, **self._config.select(data="data"))
        self.decoder = Decoder(n_patches=self.n_patches, d_patch=self.d_patch, **self._config.select(data="data"))

        self.device = device

        self.opt = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay, 
            fused=torch.cuda.is_available()
        )
        
        # Learning rate scheduler with warmup + cosine decay
        self.scheduler = create_cosine_scheduler(
            optimizer=self.opt,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            min_lr=self.min_lr,
            base_lr=self.lr,
        )
   
        
        self.lpips = lpips.LPIPS(net=self.lpips_fn, verbose=False).to(self.device)
        self.lpips.eval()
        self.lpips.requires_grad_(False)

        if self.compile:
            self.train_step = torch.compile(self.train_step)

            self.encode = torch.compile(self.encode)
            self.decode = torch.compile(self.decode)

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, device: str = "cpu") -> tuple["Tokenizer", dict]:
        checkpoint = Path(checkpoint)
        assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"

        ckpt = torch.load(checkpoint.as_posix(), map_location="cpu")
        config = ckpt["config"]
        
        conf = MiniConf(config)  
        tok = cls(device=device, **conf.select("tokenizer", data="/data"))
        tok.load_state_dict(ckpt["model"], strict=True)
        
        tok.opt.load_state_dict(ckpt["opt"])
        
        # Restore scheduler state if present
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            tok.scheduler.load_state_dict(ckpt["scheduler"])
        
        return tok, dict(
            step=ckpt["step"], 
            epoch=ckpt["epoch"], 
            config=ckpt["config"],
            best_val_loss=ckpt.get("best_val_loss", float('inf'))
        )

    def save_checkpoint(self, path: Path, step: int = 0, epoch: int = 0, best_val_loss: float = float('inf'), full_config: dict = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use full_config if provided, otherwise fall back to self._conf
        if full_config is not None:
            config = full_config
        elif hasattr(self, '_conf') and self._conf is not None:
            config = self._conf.asdict() if hasattr(self._conf, 'asdict') else self._conf
        else:
            config = {}
        
        obj = {
            "step": step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model": self.state_dict(),
            "opt": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": config,
        }
        
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    def forward(self, patches_btnd: torch.Tensor):
        z, (mae_mask, keep_prob) = self.encoder(patches_btnd)
        pred = self.decoder(z)
        return pred, mae_mask, keep_prob    
    
    @torch.no_grad()
    def encode(self, x : torch.Tensor):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            patches = temporal_patchify(x, self.num_patches)
            z, _ = self.encoder(patches)
        return z

    @torch.no_grad()
    def decode(self, z : torch.Tensor):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z = self.decoder(z)
        return z


    def train_step(self, x: torch.Tensor, accumulate: bool = False):
        """
        Single training step with optional gradient accumulation.
        
        Args:
            x: Input tensor (B,T,C,H,W) - float [0,1]
            accumulate: If True, skip optimizer step (for gradient accumulation)
        """
        patches = temporal_patchify(x, self.num_patches)
 
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred, mae_mask, keep_prob = self(patches)

            # Keep loss computation in autocast for memory efficiency
            mse = recon_loss_from_mae(pred, patches, mae_mask)

            lp = lpips_on_mae_recon(
                self.lpips, pred, patches, mae_mask,
                H=self.H, W=self.W, C=self.C, patch=self.num_patches,
                subsample_frac=self.lpips_frac
            )
            loss = mse + self.lpips_weight * lp

        loss.backward()
        
        if not accumulate:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True) 

        return loss, mse, lp, keep_prob, mae_mask

def pack_bottleneck_to_spatial(z_btLd: torch.Tensor, *, n_spatial: int, k: int) -> torch.Tensor:
    """
    z: (B,T,L,D_b) where L == n_spatial * k
    -> (B,T,n_spatial,D_b*k)
    """
    B, T, L, D = z_btLd.shape
    assert L == n_spatial * k, f"L={L} must equal n_spatial*k={n_spatial*k}"
    return z_btLd.view(B, T, n_spatial, k * D)


def unpack_spatial_to_bottleneck(z_btSd: torch.Tensor, *, k: int) -> torch.Tensor:
    """
    z: (B,T,n_spatial,D_b*k) -> (B,T,n_spatial*k,D_b)
    """
    B, T, S, DK = z_btSd.shape
    assert DK % k == 0, f"D={DK} must be divisible by k={k}"
    D = DK // k
    return z_btSd.view(B, T, S * k, D)


class ActionEncoder(nn.Module):
    """
    Continuous actions in [-1,1], shape (B,T,A) -> token (B,T,1,D).
    If actions is None (unlabeled pretrain), emits a learned base token.
    """
    def __init__(self, d_model: int, action_dim: int = 16, hidden_mult: float = 2.0):
        super().__init__()
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)

        hidden = int(self.d_model * hidden_mult)
        self.base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.base, std=0.02)

        self.fc1 = nn.Linear(self.action_dim, hidden)
        self.fc2 = nn.Linear(hidden, self.d_model)

        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        actions: Optional[torch.Tensor],                 # (B,T,A) or None
        *,
        batch_time_shape: Optional[Tuple[int,int]] = None,
        act_mask: Optional[torch.Tensor] = None,         # (B,T,A) or (A,)
        as_tokens: bool = True,
    ) -> torch.Tensor:
        if actions is None:
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = self.base.view(1, 1, -1).expand(B, T, -1)
        else:
            x = actions
            if act_mask is not None:
                x = x * act_mask
            x = x.clamp(-1, 1)
            out = self.fc2(F.silu(self.fc1(x))) + self.base.view(1, 1, -1)

        return out[:, :, None, :] if as_tokens else out


class TaskEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_patches : int,
        n_agent: int = 1,
        use_ids: bool = True,
        n_tasks: int = 128,
        d_task: int = 64,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_agent = int(n_agent)
        self.use_ids = bool(use_ids)
        self.n_tasks = int(n_tasks)
        self.d_task = int(d_task)

        if self.use_ids:
            self.task_table = nn.Embedding(self.n_tasks, self.d_model)
        else:
            self.task_proj = nn.Linear(self.d_task, self.d_model)


        self.agent_base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.agent_base, std=0.02)

    def forward(self, task: torch.Tensor, *, B: int, T: int) -> torch.Tensor:
        if self.use_ids:
            emb = self.task_table(task.to(torch.long))  # (B,D)
        else:
            emb = self.task_proj(task)                  # (B,D)

        x = emb + self.agent_base.view(1, -1)          # (B,D)
        return x[:, None, None, :].expand(B, T, self.n_agent, self.d_model)


def _emax_from_kmax(k_max: int) -> int:
    emax = int(round(math.log2(k_max)))
    assert (1 << emax) == k_max, "k_max must be power of two"
    return emax


def _sample_step_excluding_dmin(device: torch.device, B: int, T: int, k_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    emax = _emax_from_kmax(k_max)
    # step_idx in [0, emax) i.e. excludes emax (d_min)
    step_idx = torch.randint(low=0, high=max(1, emax), size=(B, T), device=device, dtype=torch.long)
    d = 1.0 / (1 << step_idx).to(torch.float32)
    return d, step_idx


def _sample_tau_for_step(device: torch.device, B: int, T: int, k_max: int, step_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # K = 2^step_idx
    K = (1 << step_idx).to(torch.long)  # (B,T)
    u = torch.rand((B, T), device=device, dtype=torch.float32)
    j_idx = torch.floor(u * K.to(torch.float32)).to(torch.long)  # (B,T) in [0,K)
    tau = j_idx.to(torch.float32) / K.to(torch.float32)          # (B,T)
    scale = torch.div(torch.tensor(k_max, device=device), K, rounding_mode="floor")  # (B,T)
    tau_idx = j_idx * scale                                      # (B,T) <= k_max-1
    return tau, tau_idx


@configclass
class Dynamics(nn.Module):
    d_model: int = config_field("d_model")
    n_heads: int = config_field("n_heads")
    depth: int = config_field("depth")
    n_register: int = config_field("n_register")
    n_agent: int = config_field("n_agent")
    k_max: int = config_field("k_max")
    dropout: float = config_field("dropout")
    mlp_ratio: float = config_field("mlp_ratio")
    time_every: int = config_field("time_every")
    space_mode: str = config_field("space_mode")

    # Optimizer config (like Tokenizer)
    lr: float = config_field("optim/lr")
    weight_decay: float = config_field("optim/weight_decay")
    grad_clip: float = config_field("optim/grad_clip")
    max_steps: int = config_field("optim/max_steps")
    warmup_steps: int = config_field("optim/warmup_steps")
    min_lr: float = config_field("optim/min_lr")
    
    packing_factor: int = config_field("packing_factor")
    self_fraction: float = config_field("self_fraction")
    bootstrap_start: int = config_field("bootstrap_start")
    
    compile: bool = config_field("training/compile")

    def __init__(self, *, tokenizer: Tokenizer, device: str):
        super().__init__()

        self.tokenizer = tokenizer
        self.device = device

        H = tokenizer.H
        W = tokenizer.W
        C = tokenizer.C

        patch = tokenizer.num_patches
        n_latents = tokenizer.decoder.n_latents
        d_bottleneck = tokenizer.decoder.d_bottleneck


        assert H % patch == 0 and W % patch == 0
        assert n_latents % self.packing_factor == 0
        self.n_spatial = n_latents // self.packing_factor
        self.d_spatial = d_bottleneck * self.packing_factor

        assert self.d_spatial % d_bottleneck == 0
        
        self.spatial_proj = nn.Linear(self.d_spatial, self.d_model)
        self.register_tokens = nn.Parameter(torch.empty(self.n_register, self.d_model))
        nn.init.normal_(self.register_tokens, std=0.02)

        self.action_encoder = ActionEncoder(d_model=self.d_model, action_dim=16)

        self.num_step_bins = int(math.log2(self.k_max)) + 1
        self.step_embed = nn.Embedding(self.num_step_bins, self.d_model)
        self.signal_embed = nn.Embedding(self.k_max + 1, self.d_model)

        segments = [
            (Modality.ACTION, 1),
            (Modality.SHORTCUT_SIGNAL, 1),
            (Modality.SHORTCUT_STEP, 1),
            (Modality.SPATIAL, self.n_spatial),
            (Modality.REGISTER, self.n_register),
        ]

        if self.n_agent > 0:
            segments.append((Modality.AGENT, self.n_agent))

        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        sl = self.layout.slices()
        self.spatial_slice = sl[Modality.SPATIAL]
        self.agent_slice = sl.get(Modality.AGENT, slice(0, 0))
        modality_ids = self.layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            n_latents=0,
            modality_ids=modality_ids,
            space_mode=self.space_mode,
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
            latents_only_time=False,
        )

        self.flow_x_head = nn.Linear(self.d_model, self.d_spatial)
        nn.init.zeros_(self.flow_x_head.weight)
        nn.init.zeros_(self.flow_x_head.bias)

        # Create optimizer (like Tokenizer)
        self.opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=torch.cuda.is_available()
        )

        # Learning rate scheduler with warmup + cosine decay
        self.scheduler = create_cosine_scheduler(
            optimizer=self.opt,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            min_lr=self.min_lr,
            base_lr=self.lr,
        )

        # Compile if enabled (like Tokenizer)
        if self.compile:
            self.train_step = torch.compile(self.train_step)

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, tokenizer: Tokenizer, device: str = "cpu") -> tuple["Dynamics", dict]:
        checkpoint = Path(checkpoint)
        assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"

        ckpt = torch.load(checkpoint.as_posix(), map_location="cpu")
        config = ckpt["config"]
        
        from miniconf import MiniConf
        conf = MiniConf(config)
        
        # Handle both full project config (with "dynamics" key) and legacy dynamics-only config
        if "dynamics" in config:
            # Full project config - select dynamics subsection
            dyn = cls(tokenizer=tokenizer, device=device, **conf.select("dynamics"))
        else:
            # Legacy: dynamics-only config (data embedded in dynamics config)
            dyn = cls(tokenizer=tokenizer, device=device, **conf.select())
        
        dyn.load_state_dict(ckpt["dynamics"], strict=True)
        
        dyn.opt.load_state_dict(ckpt["opt"])
        
        # Restore scheduler state if present
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            dyn.scheduler.load_state_dict(ckpt["scheduler"])
        
        return dyn, dict(
            step=ckpt["step"], 
            epoch=ckpt["epoch"], 
            config=ckpt["config"],
            best_val_loss=ckpt.get("best_val_loss", float('inf'))
        )

    def save_checkpoint(self, path: Path, step: int = 0, epoch: int = 0, best_val_loss: float = float('inf'), full_config: dict = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use full_config if provided, otherwise fall back to self._conf
        if full_config is not None:
            config = full_config
        elif hasattr(self, '_conf') and self._conf is not None:
            config = self._conf.asdict() if hasattr(self._conf, 'asdict') else self._conf
        else:
            config = {}
        
        obj = {
            "step": step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "dynamics": self.state_dict(),
            "opt": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": config,
        }
        
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    def forward(
        self,
        actions: Optional[torch.Tensor],          # (B,T,16) or None
        step_idxs: torch.Tensor,                  # (B,T)
        signal_idxs: torch.Tensor,                # (B,T)
        packed_enc_tokens: torch.Tensor,          # (B,T,n_spatial,d_spatial)
        *,
        act_mask: Optional[torch.Tensor] = None,  # (B,T,16) or (16,) or None
        agent_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = packed_enc_tokens.shape[:2]

        spatial_tokens = self.spatial_proj(packed_enc_tokens)  # (B,T,n_spatial,d_model)

        action_tokens = self.action_encoder(
            actions,
            batch_time_shape=(B, T),
            act_mask=act_mask,
            as_tokens=True,
        )  # (B,T,1,d_model)

        reg = self.register_tokens.view(1, 1, self.n_register, self.d_model).expand(B, T, -1, -1)

        step_tok = self.step_embed(step_idxs.to(torch.long))[:, :, None, :]
        sig_tok = self.signal_embed(signal_idxs.to(torch.long))[:, :, None, :]

        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = torch.zeros((B, T, self.n_agent, self.d_model), device=spatial_tokens.device, dtype=spatial_tokens.dtype)
            toks = [action_tokens, sig_tok, step_tok, spatial_tokens, reg, agent_tokens]
        else:
            toks = [action_tokens, sig_tok, step_tok, spatial_tokens, reg]

        tokens = torch.cat(toks, dim=2)  # (B,T,S,D)
        tokens = add_sinusoidal_positions(tokens)
        x = self.transformer(tokens)

        spatial_out = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial_out)  # (B,T,n_spatial,d_spatial)

        h_t = None
        if self.n_agent > 0:
            h_t = x[:, :, self.agent_slice, :]   # (B,T,n_agent,d_model)

        return x1_hat, h_t

    def train_step(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        act_mask: torch.Tensor,
        step: int,
        accumulate: bool = False,
    ) -> dict:
        """
        Single training step for dynamics model. Designed to be torch.compile compatible.
        
        Args:
            frames: Input frames (B,T,C,H,W)
            actions: Actions (B,T,A) or None
            act_mask: Action mask (B,T,A) or (A,) or None
            step: Current training step (for bootstrap scheduling)
            accumulate: If True, skip optimizer step (for gradient accumulation)
            
        Returns:
            Dictionary of loss values
        """
        # Frozen encoder -> packed spatial tokens z1
        with torch.no_grad():
            patches = temporal_patchify(frames, self.tokenizer.num_patches)
            z_btLd, _ = self.tokenizer.encoder(patches)
            z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=self.n_spatial, k=self.packing_factor)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            device = z1.device
            B, T = z1.shape[:2]
            B_self = int(round(self.self_fraction * B))
            B_self = max(0, min(B - 1, B_self))
            B_emp = B - B_self
            emax = _emax_from_kmax(self.k_max)

            # action mask slices
            act_mask_full = act_mask
            act_mask_self = None if act_mask_full is None else act_mask_full[B_emp:]

            # step idx: empirical rows are finest (d_min), self rows sample coarser
            step_idx_emp = torch.full((B_emp, T), emax, device=device, dtype=torch.long)
            if B_self > 0:
                d_self, step_idx_self = _sample_step_excluding_dmin(device, B_self, T, self.k_max)
                step_idx_full = torch.cat([step_idx_emp, step_idx_self], dim=0)
            else:
                d_self = torch.zeros((0, T), device=device, dtype=torch.float32)
                step_idx_self = torch.zeros((0, T), device=device, dtype=torch.long)
                step_idx_full = step_idx_emp

            # sigma/tau per row/time
            sigma_full, sigma_idx_full = _sample_tau_for_step(device, B, T, self.k_max, step_idx_full)
            sigma_emp = sigma_full[:B_emp]
            sigma_self = sigma_full[B_emp:]
            sigma_idx_self = sigma_idx_full[B_emp:]

            # Corrupt inputs
            z0_full = torch.randn_like(z1)
            z_tilde_full = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[..., None, None] * z1
            z_tilde_self = z_tilde_full[B_emp:]

            # Weights (0.9 * sigma + 0.1 gives higher weight to higher noise levels)
            w_emp = 0.9 * sigma_emp + 0.1
            w_self = 0.9 * sigma_self + 0.1

            # Main forward
            z1_hat_full, _ = self(actions, step_idx_full, sigma_idx_full, z_tilde_full, act_mask=act_mask_full, agent_tokens=None)
            z1_hat_emp = z1_hat_full[:B_emp]
            z1_hat_self = z1_hat_full[B_emp:]

            flow_per = (z1_hat_emp.float() - z1[:B_emp].float()).pow(2).mean(dim=(2, 3))  # (B_emp,T)
            loss_emp = (flow_per * w_emp).mean()

            boot_mse = torch.zeros((), device=device, dtype=torch.float32)
            loss_self = torch.zeros((), device=device, dtype=torch.float32)

            do_boot = (B_self > 0) and (step >= self.bootstrap_start)
            if do_boot:
                d_half = d_self / 2.0
                step_idx_half = step_idx_self + 1
                sigma_plus = sigma_self + d_half
                sigma_idx_plus = sigma_idx_self + (torch.tensor(self.k_max, device=device, dtype=torch.float32) * d_half).to(torch.long)

                z1_hat_half1, _ = self(actions[B_emp:] if actions is not None else None, step_idx_half, sigma_idx_self, z_tilde_self, act_mask=act_mask_self, agent_tokens=None)
                b_prime = (z1_hat_half1.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
                z_prime = z_tilde_self.float() + b_prime * d_half[..., None, None]

                z1_hat_half2, _ = self(actions[B_emp:] if actions is not None else None, step_idx_half, sigma_idx_plus, z_prime.to(z_tilde_self.dtype), act_mask=act_mask_self, agent_tokens=None)
                b_doubleprime = (z1_hat_half2.float() - z_prime.float()) / (1.0 - sigma_plus).clamp_min(1e-6)[..., None, None]

                vhat_sigma = (z1_hat_self.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
                vbar_target = ((b_prime + b_doubleprime) / 2.0).detach()

                boot_per = (1.0 - sigma_self).pow(2) * (vhat_sigma - vbar_target).pow(2).mean(dim=(2, 3))  # (B_self,T)
                loss_self = (boot_per * w_self).mean()
                boot_mse = boot_per.mean()

            # Combine losses
            loss = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B

        loss.backward()

        if not accumulate:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        return {
            "loss": loss.detach(),
            "flow_mse": flow_per.mean().detach(),
            "bootstrap_mse": boot_mse.detach(),
            "loss_emp": loss_emp.detach(),
            "loss_self": loss_self.detach(),
            "sigma_mean": sigma_full.mean().detach(),
        }

def recon_loss_from_mae(pred : torch.Tensor, target : torch.Tensor, mask : torch.Tensor):
    # Compute masked squared error without cloning pred
    diff_sq = (pred - target).square_()
    diff_sq.mul_(mask)

    denom = mask.sum().float().clamp_min_(1.0) * diff_sq.shape[-1]

    return diff_sq.sum(dtype=torch.float32) / denom


def lpips_on_mae_recon(
    lpips_fn,
    pred_btnd: torch.Tensor,
    target_btnd: torch.Tensor,
    mae_mask_btNp1: torch.Tensor,
    *,
    H: int, W: int, C: int, patch: int,
    subsample_frac: float = 1.0,
) -> torch.Tensor:
    
    recon_masked_btnd = torch.where(mae_mask_btNp1, pred_btnd, target_btnd)
    recon = temporal_unpatchify(recon_masked_btnd,  H, W, C, patch)
    tgt   = temporal_unpatchify(target_btnd,        H, W, C, patch)

    if subsample_frac < 1.0:
        B, T = recon.shape[:2]
        step = max(1, int(1.0 / subsample_frac))
        recon = recon[:, ::step]
        tgt   = tgt[:, ::step]

    recon = (recon.clamp(0, 1) * 2.0 - 1.0)
    tgt   = (tgt.clamp(0, 1)   * 2.0 - 1.0)

    B, T = recon.shape[:2]
    recon = recon.reshape(B * T, C, H, W)
    tgt   = tgt.reshape(B * T, C, H, W)
    
    lp = lpips_fn(recon, tgt)

    return lp.mean()
