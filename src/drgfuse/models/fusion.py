"""
DRGFuse core (fusion-side only).

This module implements an alignment-guided, dual reliability-gated fusion core
for CT/WSI feature fusion as described in the accompanying manuscript.

Scope
- This file only contains the fusion-side network.
- CT/WSI feature extraction (segmentation, CT encoder, WSI tiling, UNI, etc.)
  is intentionally out of scope and should be performed offline.

Inputs per sample (two modalities)
- tokens (optional): (B, N, D_token)
- global embedding: (B, D_global)
- unimodal logit: (B,) or (B, 1)
- quality vector: (B, Q)

Outputs
- mismatch score m in [0, 1]
- pre-gate alpha = 1 - m
- fused representation z_x and interaction expert logit y_x
- post-gate routing weights over {CT, WSI, Fuse}
- final logit y and probability p = sigmoid(y)

Notes
- This implementation uses a lightweight coupled cross-attention + token mixer
  as a practical approximation of a cross-sequence interaction block.
- Training-only components (alignment guards, co-teaching, low-FPR objective)
  are implemented outside this file and are not required at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Optional dependency for the manuscript's sequence module (Mamba).
try:
    from mamba_ssm import Mamba  # type: ignore
except Exception:  # pragma: no cover
    Mamba = None  # fallback to Transformer



@dataclass
class FusionInputs:
    ct_tokens: Optional[torch.Tensor]
    ct_global: torch.Tensor
    ct_logit: torch.Tensor
    ct_quality: torch.Tensor

    wsi_tokens: Optional[torch.Tensor]
    wsi_global: torch.Tensor
    wsi_logit: torch.Tensor
    wsi_quality: torch.Tensor

    ct_token_mask: Optional[torch.Tensor] = None  # (B, N_ct) bool
    wsi_token_mask: Optional[torch.Tensor] = None  # (B, N_wsi) bool


@dataclass
class FusionOutputs:
    mismatch_score: torch.Tensor  # (B, 1)
    mismatch_embed: torch.Tensor  # (B, D_delta)
    alpha: torch.Tensor           # (B, 1)
    z_x: torch.Tensor             # (B, D_model)
    y_x: torch.Tensor             # (B,)
    gate_weights: torch.Tensor    # (B, 3) -> [w_fuse, w_ct, w_wsi]
    y: torch.Tensor               # (B,)
    p: torch.Tensor               # (B,)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _as_1d_logit(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2 and x.shape[1] == 1:
        return x[:, 0]
    if x.dim() == 1:
        return x
    raise ValueError(f"Expected logit with shape (B,) or (B,1), got {tuple(x.shape)}")


def _default_token_mask(tokens: torch.Tensor) -> torch.Tensor:
    b, n, _ = tokens.shape
    return torch.ones((b, n), dtype=torch.bool, device=tokens.device)


class AttentivePooling(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.query.unsqueeze(0).unsqueeze(1)  # (1,1,D)
        k = self.proj(tokens)                     # (B,N,D)
        attn = torch.matmul(k, q.transpose(-1, -2)).squeeze(-1)  # (B,N)
        attn = attn.masked_fill(~mask, float("-inf"))
        w = F.softmax(attn, dim=1)
        w = self.dropout(w)
        z = torch.matmul(w.unsqueeze(1), tokens).squeeze(1)       # (B,D)
        return z


class DisparityReliabilityEstimator(nn.Module):
    """Mismatch estimation as in the manuscript (Eq. 2).

    Inputs:
      - z_ct: (B, D_ct_global)
      - z_wsi: (B, D_wsi_global)

    Outputs:
      - m: (B, 1) mismatch score in [0,1]
      - z_delta: (B, d_delta) discrepancy embedding
    """

    def __init__(
        self,
        d_ct: int,
        d_wsi: int,
        d_model: int = 256,
        d_delta: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ct_proj = nn.Linear(d_ct, d_model)
        self.wsi_proj = nn.Linear(d_wsi, d_model)

        # [z_ct; z_wsi; (z_ct - z_wsi); (z_ct ⊙ z_wsi)]
        in_dim = d_model * 4
        self.backbone = MLP(in_dim, hidden_dim, hidden_dim, n_layers=2, dropout=dropout)
        self.to_delta = nn.Linear(hidden_dim, d_delta)
        self.to_m = nn.Linear(hidden_dim, 1)

    def forward(self, z_ct: torch.Tensor, z_wsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zc = self.ct_proj(z_ct)
        zw = self.wsi_proj(z_wsi)
        feat = torch.cat([zc, zw, (zc - zw), zc * zw], dim=-1)
        h = self.backbone(feat)
        z_delta = self.to_delta(h)
        m = torch.sigmoid(self.to_m(h))  # (B,1)
        return m, z_delta


class PreGate(nn.Module):
    def __init__(self, temperature: float = 1.0, alpha_min: float = 0.0, alpha_max: float = 1.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        if self.temperature != 1.0:
            m = torch.sigmoid(torch.logit(m.clamp(1e-6, 1 - 1e-6)) / self.temperature)
        alpha = (1.0 - m).clamp(self.alpha_min, self.alpha_max)
        return alpha


class DepthwiseTokenMixer(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.dw = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.pw = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)         # (B,D,N)
        y = self.dw(x_t)
        y = self.pw(F.gelu(y))
        y = y.transpose(1, 2)           # (B,N,D)
        return self.dropout(y)


class CoupledCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.ct_norm = nn.LayerNorm(d_model)
        self.wsi_norm = nn.LayerNorm(d_model)

        self.ct_xattn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.wsi_xattn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ct_mixer = DepthwiseTokenMixer(d_model, dropout=dropout)
        self.wsi_mixer = DepthwiseTokenMixer(d_model, dropout=dropout)

        self.ct_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.wsi_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
        # MultiheadAttention expects True for positions that should be ignored.
        return ~mask

    def forward(
        self,
        ct: torch.Tensor,
        wsi: torch.Tensor,
        alpha: torch.Tensor,
        ct_mask: torch.Tensor,
        wsi_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a = alpha.unsqueeze(-1)  # (B,1,1)

        ct_q = self.ct_norm(ct)
        wsi_kv = self.wsi_norm(wsi)
        ct_ctx, _ = self.ct_xattn(
            query=ct_q,
            key=wsi_kv,
            value=wsi_kv,
            key_padding_mask=self._key_padding_mask(wsi_mask),
            need_weights=False,
        )
        ct = ct + a * ct_ctx

        wsi_q = self.wsi_norm(wsi)
        ct_kv = self.ct_norm(ct)
        wsi_ctx, _ = self.wsi_xattn(
            query=wsi_q,
            key=ct_kv,
            value=ct_kv,
            key_padding_mask=self._key_padding_mask(ct_mask),
            need_weights=False,
        )
        wsi = wsi + a * wsi_ctx

        ct = ct + self.ct_mixer(self.ct_norm(ct))
        wsi = wsi + self.wsi_mixer(self.wsi_norm(wsi))

        ct = ct + self.ct_ffn(ct)
        wsi = wsi + self.wsi_ffn(wsi)

        ct = ct * ct_mask.to(ct.dtype).unsqueeze(-1)
        wsi = wsi * wsi_mask.to(wsi.dtype).unsqueeze(-1)
        return ct, wsi


class SequenceBackbone(nn.Module):
    """A lightweight sequence model with an optional Mamba backend.

    If `mamba_ssm` is available, we use Mamba blocks; otherwise we fall back to a
    Transformer encoder (still deterministic and easy to install).

    This keeps the released code runnable while matching the manuscript's intent.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.1, prefer_mamba: bool = True) -> None:
        super().__init__()
        self.prefer_mamba = bool(prefer_mamba)

        self._use_mamba = False
        if self.prefer_mamba and (Mamba is not None):
            self._use_mamba = True
            self.blocks = nn.ModuleList(
                [Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)]
            )
            self.norm = nn.LayerNorm(d_model)
        else:
            enc = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D); mask: (B, N) bool
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B,N,D), got {tuple(x.shape)}")
        if mask.dim() != 2:
            raise ValueError(f"Expected mask with shape (B,N), got {tuple(mask.shape)}")

        if self._use_mamba:
            # Zero-out padded tokens (Mamba has no native key padding mask).
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
            return x

        # Transformer path
        key_padding_mask = ~mask  # True means "pad" for torch Transformer
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class CoupledCrossInteraction(nn.Module):
    """Pre-gated controlled interaction via cross-modal projections + sequence update.

    This follows the manuscript's design: cross-modal projection → sequence update,
    with interaction strength modulated by alpha = 1 - m.
    """

    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj_w2c = nn.Linear(d_model, d_model, bias=False)
        self.proj_c2w = nn.Linear(d_model, d_model, bias=False)

        self.ct_seq = SequenceBackbone(d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout, prefer_mamba=True)
        self.wsi_seq = SequenceBackbone(d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout, prefer_mamba=True)

        self.ct_pool = AttentivePooling(d_model=d_model, dropout=dropout)
        self.wsi_pool = AttentivePooling(d_model=d_model, dropout=dropout)

    def forward(
        self,
        ct_tokens: torch.Tensor,
        wsi_tokens: torch.Tensor,
        *,
        alpha: torch.Tensor,
        ct_mask: torch.Tensor,
        wsi_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # alpha: (B,1)
        alpha = alpha.view(-1, 1, 1).to(dtype=ct_tokens.dtype)

        # Cross-modal injection (scaled by alpha)
        w2c = self.proj_w2c(wsi_tokens) * alpha
        c2w = self.proj_c2w(ct_tokens) * alpha

        # Concatenate injected tokens (simple and length-agnostic)
        ct_seq = torch.cat([ct_tokens, w2c], dim=1)
        wsi_seq = torch.cat([wsi_tokens, c2w], dim=1)

        ct_seq_mask = torch.cat([ct_mask, wsi_mask], dim=1)
        wsi_seq_mask = torch.cat([wsi_mask, ct_mask], dim=1)

        ct_seq = self.ct_seq(ct_seq, ct_seq_mask)
        wsi_seq = self.wsi_seq(wsi_seq, wsi_seq_mask)

        # Pool to per-sample vectors
        z_ct = self.ct_pool(ct_seq, ct_seq_mask)
        z_wsi = self.wsi_pool(wsi_seq, wsi_seq_mask)
        return z_ct, z_wsi


class PostGateMoE(nn.Module):
    """Post-gated routing (MoE-style) using only (q_ct, q_wsi, m, z_delta).

    Matches Eq. (4) in the manuscript.
    Weight order: [w_fuse, w_ct, w_wsi].
    """

    def __init__(
        self,
        q_ct_dim: int,
        q_wsi_dim: int,
        d_delta: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        in_dim = q_ct_dim + q_wsi_dim + 1 + d_delta
        self.gate = MLP(in_dim, hidden_dim, 3, n_layers=2, dropout=dropout)

    def forward(
        self,
        *,
        m: torch.Tensor,
        z_delta: torch.Tensor,
        q_ct: torch.Tensor,
        q_wsi: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,  # (B,3) bool; True=available
    ) -> torch.Tensor:
        m = m.view(-1, 1)
        feat = torch.cat([q_ct, q_wsi, m, z_delta], dim=-1)
        logits = self.gate(feat)  # (B,3)

        if modality_mask is not None:
            logits = logits.masked_fill(~modality_mask, float("-inf"))

        if self.temperature != 1.0:
            logits = logits / self.temperature

        return F.softmax(logits, dim=-1)


class ResidualLogitFusion(nn.Module):
    """Final logit fusion: s = w_fuse*y_fuse + w_ct*y_ct + w_wsi*y_wsi."""

    def forward(self, y_ct: torch.Tensor, y_wsi: torch.Tensor, y_x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        y_ct = _as_1d_logit(y_ct)
        y_wsi = _as_1d_logit(y_wsi)
        y_x = _as_1d_logit(y_x)
        w_fuse, w_ct, w_wsi = w[:, 0], w[:, 1], w[:, 2]
        return w_fuse * y_x + w_ct * y_ct + w_wsi * y_wsi


class ControlledInteractionFusion(nn.Module):
    def __init__(
        self,
        d_ct_token: int,
        d_wsi_token: int,
        d_ct_global: int,
        d_wsi_global: int,
        q_ct_dim: int,
        q_wsi_dim: int,
        d_model: int = 256,
        d_delta: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ct_tok_proj = nn.Linear(d_ct_token, d_model)
        self.wsi_tok_proj = nn.Linear(d_wsi_token, d_model)

        self.mismatch = DisparityReliabilityEstimator(
            d_ct=d_ct_global,
            d_wsi=d_wsi_global,
            d_model=d_model,
            d_delta=d_delta,
            hidden_dim=max(256, d_model),
            dropout=dropout,
        )
        self.pre_gate = PreGate(temperature=1.0)

        self.interaction = CoupledCrossInteraction(d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)

        self.fuse_proj = nn.Sequential(
            nn.LayerNorm(d_model * 3 + d_delta),
            nn.Linear(d_model * 3 + d_delta, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.interaction_head = nn.Linear(d_model, 1)

        self.post_gate = PostGateMoE(
            q_ct_dim=q_ct_dim,
            q_wsi_dim=q_wsi_dim,
            d_delta=d_delta,
            hidden_dim=max(256, d_model),
            dropout=dropout,
            temperature=1.0,
        )
        self.res_fuse = ResidualLogitFusion()

        # Optional unimodal heads (used when unimodal logits are not provided).
        self.ct_unimodal_head = nn.Linear(d_ct_global, 1)
        self.wsi_unimodal_head = nn.Linear(d_wsi_global, 1)

    @staticmethod
    def _tokens_from_global(z: torch.Tensor) -> torch.Tensor:
        return z.unsqueeze(1)  # (B,1,D)

    def forward(self, inp: FusionInputs) -> FusionOutputs:
        ct_tokens = inp.ct_tokens if inp.ct_tokens is not None else self._tokens_from_global(inp.ct_global)
        wsi_tokens = inp.wsi_tokens if inp.wsi_tokens is not None else self._tokens_from_global(inp.wsi_global)

        ct_mask = inp.ct_token_mask if inp.ct_token_mask is not None else _default_token_mask(ct_tokens)
        wsi_mask = inp.wsi_token_mask if inp.wsi_token_mask is not None else _default_token_mask(wsi_tokens)

        ct_tok = self.ct_tok_proj(ct_tokens)
        wsi_tok = self.wsi_tok_proj(wsi_tokens)

        m, z_delta = self.mismatch(inp.ct_global, inp.wsi_global)
        alpha = self.pre_gate(m)

        ct_z, wsi_z = self.interaction(ct_tok, wsi_tok, alpha=alpha, ct_mask=ct_mask, wsi_mask=wsi_mask)

        zx_in = torch.cat([ct_z, wsi_z, torch.abs(ct_z - wsi_z), z_delta], dim=-1)
        z_x = self.fuse_proj(zx_in)
        y_x = self.interaction_head(z_x).squeeze(-1)

        y_ct = inp.ct_logit
        if y_ct is None:
            y_ct = self.ct_unimodal_head(inp.ct_global).squeeze(-1)

        y_wsi = inp.wsi_logit
        if y_wsi is None:
            y_wsi = self.wsi_unimodal_head(inp.wsi_global).squeeze(-1)

        w = self.post_gate(
            m=m,
            z_delta=z_delta,
            q_ct=inp.ct_quality,
            q_wsi=inp.wsi_quality,
            modality_mask=None,
        )

        y = self.res_fuse(y_ct, y_wsi, y_x, w)
        p = torch.sigmoid(y)

        return FusionOutputs(
            mismatch_score=m,
            mismatch_embed=z_delta,
            alpha=alpha,
            z_x=z_x,
            y_x=y_x,
            gate_weights=w,
            y=y,
            p=p,
        )
