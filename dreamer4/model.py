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
from einops import rearrange, repeat
from miniconf import configclass, MiniConf, config_field
import lpips

from attention import GroupedQueryAttention


# =============================================================================
# Learning Rate Scheduler
# =============================================================================


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

class MAEReplacer(nn.Module):
    def __init__(self, d_model: int, p_min: float = 0.0, p_max: float = 0.9):
        super().__init__()
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.mask_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, patches_btnd: torch.Tensor, disable_mae = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if (self.p_min == 0.0 and self.p_max == 0.0) or disable_mae:
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

@configclass
class SpaceSelfAttentionModality(nn.Module):
    n_latents: float = config_field("num_latents")
    d_model: float = config_field("latent_dim")

    def __init__(self, modality_ids: torch.Tensor, mode: str):
        super().__init__()

        self.mode = mode
        self.register_buffer("modality_ids", modality_ids.to(torch.int32), persistent=False)

        S = int(self.modality_ids.numel())
        allow = self._build_allow(S)
        attn_mask = allow.unsqueeze(0).unsqueeze(0)
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        self._use_mask = self.mode not in ("wm_agent",)

        self.attn = GroupedQueryAttention(self.d_model, **self._config.select("attention"))
    
    def _build_allow(self, S: int) -> torch.Tensor:
        device = self.modality_ids.device
        q_idx = torch.arange(S, device=device).unsqueeze(1)
        k_idx = torch.arange(S, device=device).unsqueeze(0)

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
            return torch.ones((S, S), dtype=torch.bool, device=device)

        if self.mode == "wm_agent_isolated":
            is_q_agent = (q_mod == int(Modality.AGENT))
            is_k_agent = (k_mod == int(Modality.AGENT))
            allow = torch.ones((S, S), dtype=torch.bool, device=device)
            allow_non_agent_q = ~is_q_agent
            allow = torch.where(allow_non_agent_q, ~is_k_agent, allow)
            allow = torch.where(is_q_agent, is_k_agent, allow)
            return allow

        raise ValueError(f"Unsupported mode for tokenizer/wm: {self.mode}")

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        x = rearrange(x_btSd, 'b t s d -> (b t) s d')
        mask = self.attn_mask if self._use_mask else None
        y = self.attn(x, attn_mask=mask, is_causal=False)
        return rearrange(y, '(b t) s d -> b t s d', b=B, t=T)

@configclass
class TimeSelfAttention(nn.Module):
    latents_only: bool = config_field("latents_only_time")
    n_latents: float = config_field("num_latents")
    d_model: float = config_field("latent_dim")

    def __init__(self):
        super().__init__()   
        self.attn = GroupedQueryAttention(self.d_model, **self._config.select("attention"))
   
    def forward(self, x: torch.Tensor):
        if self.latents_only:
            return self.forward_latents_only(x)
        else:
            return self.forward_latents_all(x)
        
    def forward_latents_all(self, x_btSd: torch.Tensor):
        B, T, S, D = x_btSd.shape
        x_bst = rearrange(x_btSd, 'b t s d -> (b s) t d')
        out = self.attn(x_bst, is_causal=True)
        return rearrange(out, '(b s) t d -> b t s d', b=B, s=S)

    def forward_latents_only(self, x_btSd: torch.Tensor):
        B, T, S, D = x_btSd.shape
        L = self.n_latents
        lat = rearrange(x_btSd[:, :, :L, :], 'b t l d -> (b l) t d')
        out = self.attn(lat, is_causal=True)
        out = rearrange(out, '(b l) t d -> b t l d', b=B, l=L)
        return torch.cat([out, x_btSd[:, :, L:, :]], dim=2)


@configclass
class BlockCausalLayer(nn.Module):
    d_model: int = config_field("latent_dim")
    time_every: float = config_field("time_embedding_every")
    mlp_ratio: float = config_field("mlp_ratio")
    dropout: float = config_field("dropout")
    
    def __init__(self, layer_index : int, modality_ids: torch.Tensor, space_mode: str):
        super().__init__()
        self.do_time = ((layer_index + 1) % self.time_every == 0)

        self.norm = nn.RMSNorm(self.d_model)
        
        self.space_attn = nn.Sequential(
            SpaceSelfAttentionModality(modality_ids, space_mode, **self._config.select()),
            nn.Dropout(self.dropout, True)
        )
        
        self.time_attn = nn.Identity() if not self.do_time else nn.Sequential(
            TimeSelfAttention(**self._config.select()),
            nn.Dropout(self.dropout, True),
        )
        
        self.mlp = nn.Sequential(
            MLP(self.d_model, mlp_ratio=self.mlp_ratio, dropout=self.dropout),
            nn.Dropout(self.dropout, True),
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        
        attn_out = self.space_attn(x_norm)
        
        time = self.time_attn(x_norm) if self.do_time else 0
        attn_out = attn_out + time
            
        mlp_out = self.mlp(x_norm)
        
        x = x + attn_out + mlp_out
        return x


@configclass
class BlockCausalTransformer(nn.Module):
    gradient_checkpointing : bool = True
    depth : int = config_field("num_layers")

    def __init__(
        self,
        modality_ids: torch.Tensor,
        space_mode: str,
    ):
        super().__init__()
        self.layers = nn.Sequential(*[
            BlockCausalLayer(
                layer_index=i,
                modality_ids=modality_ids,
                space_mode=space_mode,
                **self._config.select()
            )
            for i in range(self.depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            for layer in self.layers:
                x = grad_checkpoint(layer, x, use_reentrant=False)
            return x
        return self.layers(x)


@configclass
class Encoder(nn.Module):
    d_model : int = config_field("latent_dim")
    d_bottleneck : int = config_field("bottleneck_dim")
    n_latents : int = config_field("num_latents")

    mae_p_min : float = config_field("mae_prob_min")
    mae_p_max : float = config_field("mae_prob_max")


    def __init__(self, n_patches: int, d_patch: int):
        super().__init__()

        self.n_patches = n_patches
        self.d_patch = d_patch

        self.patch_proj = nn.Linear(self.d_patch, self.d_model)
        self.bottleneck_proj = nn.Linear(self.d_model, self.d_bottleneck)

        self.layout = TokenLayout(n_latents=self.n_latents, segments=((Modality.IMAGE, self.n_patches),))
        modality_ids = self.layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            modality_ids=modality_ids, space_mode="encoder",
            **self._config.select(),
        )

        self.mae = MAEReplacer(d_model=self.d_model, p_min=self.mae_p_min, p_max=self.mae_p_max)

        self.latents = nn.Parameter(torch.empty(self.n_latents, self.d_model))
        nn.init.normal_(self.latents, std=0.02)

    def forward(self, patch_tokens_btnd: torch.Tensor, disable_mae=False):
        B, T, Np, Dp = patch_tokens_btnd.shape
        
        proj = self.patch_proj(patch_tokens_btnd)
        proj_masked, mae_mask, keep_prob = self.mae(proj, disable_mae)

        lat = repeat(self.latents, 'l d -> b t l d', b=B, t=T)
        tokens = torch.cat([lat, proj_masked], dim=2)


        enc = self.transformer(tokens)
        z = torch.tanh(self.bottleneck_proj(enc[:, :, :self.n_latents, :]))
        return z, (mae_mask, keep_prob)


@configclass
class Decoder(nn.Module):

    d_model : int = config_field("latent_dim")
    d_bottleneck : int = config_field("bottleneck_dim")
    n_latents : int = config_field("num_latents")

    def __init__(self, n_patches: int, d_patch: int):
        super().__init__()

        self.n_patches = n_patches
        self._d_patch = d_patch

        self.up_proj = nn.Linear(self.d_bottleneck, self.d_model)
        self.patch_queries = nn.Parameter(torch.empty(self.n_patches, self.d_model))
        nn.init.normal_(self.patch_queries, std=0.02)

        self.patch_head = nn.Linear(self.d_model, self._d_patch)

        self.layout = TokenLayout(n_latents=self.n_latents, segments=((Modality.IMAGE, self.n_patches),))
        modality_ids = self.layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            modality_ids=modality_ids, space_mode="decoder",
            **self._config.select()
        )

    def forward(self, z_btLd: torch.Tensor) -> torch.Tensor:
        B, T, L, _ = z_btLd.shape
        
        lat = torch.tanh(self.up_proj(z_btLd))
        qry = repeat(self.patch_queries, 'p d -> b t p d', b=B, t=T)
        tokens = torch.cat([lat, qry], dim=2)

        x = self.transformer(tokens)
        x_p = x[:, :, L:, :]
        return torch.sigmoid(self.patch_head(x_p))

@configclass
class Tokenizer(nn.Module):

    num_latents : int = config_field("num_latents", ge=1)
    d_bottleneck : int = config_field("bottleneck_dim")
  
    P : int = config_field("patch_size")  # This is actually patch_size (kernel/stride)
        
    def __init__(self, C, H, W):
        super().__init__()
        
        assert H % self.P == 0 and W % self.P == 0
        self.n_patches = (H // self.P) * (W // self.P)  # number of patches
        self.d_patch = self.P * self.P * C              # patch dimension (pixels per patch)

        self.encoder = Encoder(n_patches=self.n_patches, d_patch=self.d_patch, **self._config.select())
        self.decoder = Decoder(n_patches=self.n_patches, d_patch=self.d_patch, **self._config.select())

    def forward(self, patches_btnd: torch.Tensor):
        z, (mae_mask, keep_prob) = self.encoder(patches_btnd)
        pred = self.decoder(z)
        return pred, mae_mask, keep_prob    
    
    def encode(self, x : torch.Tensor):
        patches = temporal_patchify(x, self.P)
        z, _ = self.encoder(patches)
        return z

    def decode(self, z : torch.Tensor):
        z = self.decoder(z)
        return z

def pack_bottleneck_to_spatial(z_btLd: torch.Tensor, *, n_spatial: int, k: int) -> torch.Tensor:
    """
    Pack encoded states Batch x Temporal x N_Latents x D_bottelneck to Batch x Temporal x N_Spatial x D_Spatial.
    Basically we tokenized a temporal set of images into N_Latents of D_bottelneck latents
    for the sake of performance we pack 2 or more (k) into one token for our dynamics model.
    """
    return rearrange(z_btLd, 'b t (n k) d -> b t n (k d)', n=n_spatial, k=k)


def unpack_spatial_to_bottleneck(z_btSd: torch.Tensor, *, k: int) -> torch.Tensor:
    """
    Look at pack_bottleneck_to_spatial its just the reverse.
    """
    return rearrange(z_btSd, 'b t n (k d) -> b t (n k) d', k=k)


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
        actions: Optional[torch.Tensor],
        *,
        batch_time_shape: Optional[Tuple[int,int]] = None,
        act_mask: Optional[torch.Tensor] = None,
        as_tokens: bool = True,
    ) -> torch.Tensor:
        if actions is None:
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = repeat(self.base, 'd -> b t d', b=B, t=T)
        else:
            x = actions * act_mask if act_mask is not None else actions
            x = x.clamp(-1, 1)
            out = self.fc2(F.silu(self.fc1(x))) + self.base

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
        emb = self.task_table(task) if self.use_ids else self.task_proj(task)
        x = emb + self.agent_base
        return repeat(x, 'b d -> b t n d', t=T, n=self.n_agent)


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
    d_model: int = config_field("latent_dim")
    depth: int = config_field("num_layers")
    n_register: int = config_field("n_register")
    n_agent: int = config_field("n_agent")
    k_max: int = config_field("k_max")
    dropout: float = config_field("dropout")
    mlp_ratio: float = config_field("mlp_ratio")
    time_every: int = config_field("time_embedding_every")
    space_mode: str = config_field("space_mode")


    def __init__(self, *, tokenizer: Tokenizer, device: str):
        super().__init__()

        self.tokenizer = tokenizer
        self.device = device

        n_latents = tokenizer.decoder.n_latents
        d_bottleneck = tokenizer.decoder.d_bottleneck


        assert n_latents % self.packing_factor == 0
        self.n_spatial = n_latents // self.packing_factor
        self.d_spatial = d_bottleneck * self.packing_factor

        assert self.d_spatial % d_bottleneck == 0
        
        self.spatial_proj = nn.Linear(self.d_spatial, self.d_model)

        # ?What are these register tokens for
        self.register_tokens = nn.Parameter(torch.empty(self.n_register, self.d_model))
        nn.init.normal_(self.register_tokens, std=0.02)

        self.action_encoder = ActionEncoder(d_model=self.d_model, action_dim=16)

        self.num_step_bins = int(math.log2(self.k_max)) + 1
        self.step_embed = nn.Embedding(self.num_step_bins, self.d_model)
        self.signal_embed = nn.Embedding(self.k_max + 1, self.d_model)

        """
        Okay so what are these segments and
        """
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
            modality_ids=modality_ids,
            space_mode=self.space_mode,
            **self._config.select()
        )

        self.flow_x_head = nn.Linear(self.d_model, self.d_spatial)
        nn.init.normal_(self.flow_x_head.weight, std=0.001)
        nn.init.zeros_(self.flow_x_head.bias)


    @classmethod
    def from_checkpoint(cls, checkpoint: Path, tokenizer: Tokenizer, device: str = "cpu") -> tuple["Dynamics", dict]:
        checkpoint = Path(checkpoint)
        assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"

        ckpt = torch.load(checkpoint.as_posix(), map_location="cpu")
        config = ckpt["config"]
        
        from miniconf import MiniConf
        conf = MiniConf(config)
        
        dyn = cls(tokenizer=tokenizer, device=device, **conf.select("dynamics"))
        dyn.load_state_dict(ckpt["dynamics"], strict=True)
        
        # Move model to device BEFORE loading optimizer state
        # This ensures optimizer state tensors are created on the correct device
        # (required for fused AdamW which expects all tensors on the same device)
        dyn.to(device)
        
        dyn.opt.load_state_dict(ckpt["opt"])
        
        # Move optimizer state tensors to device (they were loaded on CPU)
        for state in dyn.opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
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
        actions: Optional[torch.Tensor],
        step_idxs: torch.Tensor,
        signal_idxs: torch.Tensor,
        packed_enc_tokens: torch.Tensor,
        *,
        act_mask: Optional[torch.Tensor] = None,
        agent_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = packed_enc_tokens.shape[:2]

        spatial_tokens = self.spatial_proj(packed_enc_tokens)

        action_tokens = self.action_encoder(
            actions,
            batch_time_shape=(B, T),
            act_mask=act_mask,
            as_tokens=True,
        )

        reg = repeat(self.register_tokens, 'r d -> b t r d', b=B, t=T)

        step_tok = rearrange(self.step_embed(step_idxs), 'b t d -> b t 1 d')
        sig_tok = rearrange(self.signal_embed(signal_idxs), 'b t d -> b t 1 d')

        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = torch.zeros((B, T, self.n_agent, self.d_model), device=spatial_tokens.device, dtype=spatial_tokens.dtype)
            toks = [action_tokens, sig_tok, step_tok, spatial_tokens, reg, agent_tokens]
        else:
            toks = [action_tokens, sig_tok, step_tok, spatial_tokens, reg]

        tokens = torch.cat(toks, dim=2)
        x = self.transformer(tokens)

        spatial_out = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial_out)

        h_t = None
        if self.n_agent > 0:
            h_t = x[:, :, self.agent_slice, :]

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
