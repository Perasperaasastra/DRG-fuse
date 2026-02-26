"""
Training losses and regularizers.

Main classification loss
- Weighted BCE with logits.

Low-FPR objective (approximate)
- A batch-wise surrogate that emphasizes separating positives from the
  highest-scoring negatives, approximating the low-FPR region.

Alignment guards (training-only, optional)
- Token-level Sinkhorn OT distance (entropic regularized OT).
- Global-level multi-kernel RBF MMD.

Gate regularization (optional)
- Entropy regularization (encourage non-collapsed routing).
- Load balancing regularization.

Co-teaching (training strategy, optional)
- A mismatch-aware small-loss sample selection strategy for noisy labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    bce: float = 1.0
    low_fpr: float = 1.0
    ot: float = 0.1
    mmd: float = 0.1
    gate_entropy: float = 0.001
    gate_balance: float = 0.001


@dataclass(frozen=True)
class CoTeachConfig:
    enable: bool = False
    noise_rate: float = 0.2
    warmup_epochs: int = 5
    mismatch_gamma: float = 1.0


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    logits: (B,)
    targets: (B,) in {0,1}
    pos_weight: scalar, typically n_neg / n_pos computed on train split
    sample_weight: optional per-sample weights (B,)
    """
    logits = logits.view(-1)
    targets = targets.view(-1).to(dtype=logits.dtype)

    w = torch.tensor([float(pos_weight)], dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=w, reduction="none")

    if sample_weight is not None:
        sw = sample_weight.view(-1).to(device=logits.device, dtype=logits.dtype)
        loss = loss * sw

    return loss.mean()


def low_fpr_pairwise_surrogate(
    logits: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """
    Approximate low-FPR optimization by focusing on the highest-scoring negatives.

    Steps
    - Select top-k negatives where k = ceil(beta * N_neg) (at least 1 if N_neg>0).
    - Compute pairwise softplus loss between each positive and selected negatives:
        L = mean_{pos, hard_neg} softplus(-(s_pos - s_neg))
    """
    logits = logits.view(-1)
    targets = targets.view(-1).to(dtype=torch.long)

    pos = logits[targets == 1]
    neg = logits[targets == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)

    beta = float(beta)
    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must be in (0, 1].")

    n_neg = int(neg.numel())
    k = max(1, int(math.ceil(beta * n_neg)))

    # Non-differentiable selection is performed on detached scores.
    _, idx = torch.topk(neg.detach(), k=k, largest=True)
    hard_neg = neg[idx]  # keep gradient for selected negatives

    # Pairwise logistic loss
    diff = pos.unsqueeze(1) - hard_neg.unsqueeze(0)
    return F.softplus(-diff).mean()


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    xy = x @ y.transpose(0, 1)
    dist2 = (x2 + y2 - 2 * xy).clamp_min(0.0)
    return torch.exp(-gamma * dist2)


def mmd_rbf_multi(
    x: torch.Tensor,
    y: torch.Tensor,
    gammas: Tuple[float, ...] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """
    Multi-kernel RBF MMD between two sets of embeddings.
    x: (B, D), y: (B, D)
    """
    if x.numel() == 0 or y.numel() == 0:
        return x.new_tensor(0.0)

    k_xx = 0.0
    k_yy = 0.0
    k_xy = 0.0
    for g in gammas:
        k_xx = k_xx + _rbf_kernel(x, x, gamma=g)
        k_yy = k_yy + _rbf_kernel(y, y, gamma=g)
        k_xy = k_xy + _rbf_kernel(x, y, gamma=g)

    mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return mmd


def sinkhorn_ot_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    x_mask: torch.Tensor,
    y_mask: torch.Tensor,
    eps: float = 0.05,
    n_iters: int = 30,
    cost: str = "cosine",
) -> torch.Tensor:
    """
    Entropic regularized OT distance (Sinkhorn).

    x: (B, N, D), y: (B, M, D)
    x_mask, y_mask: (B, N)/(B, M) bool, True = valid
    """
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("x and y must be 3D tensors (B,N,D) and (B,M,D).")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Batch size mismatch.")
    if x.shape[2] != y.shape[2]:
        raise ValueError("Feature dimension mismatch.")

    b, n, d = x.shape
    m = y.shape[1]
    x_mask = x_mask.to(dtype=torch.bool, device=x.device)
    y_mask = y_mask.to(dtype=torch.bool, device=x.device)

    # Normalize tokens for cosine cost.
    if cost == "cosine":
        x_n = F.normalize(x, p=2, dim=-1)
        y_n = F.normalize(y, p=2, dim=-1)
        sim = x_n @ y_n.transpose(1, 2)  # (B,N,M)
        c = (1.0 - sim).clamp_min(0.0)
    elif cost == "l2":
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1).unsqueeze(1)
        xy = x @ y.transpose(1, 2)
        c = (x2 + y2 - 2.0 * xy).clamp_min(0.0)
    else:
        raise ValueError("Unsupported cost type.")

    # Mask invalid positions by setting cost to a large value.
    big = c.max().detach() + 1.0
    c = c.masked_fill(~(x_mask.unsqueeze(-1) & y_mask.unsqueeze(1)), big)

    # Uniform marginals over valid tokens.
    a = x_mask.to(x.dtype)
    b_mask = y_mask.to(x.dtype)
    a = a / a.sum(dim=1, keepdim=True).clamp_min(1.0)
    b_mask = b_mask / b_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    # Sinkhorn iterations in log domain for stability.
    # K = exp(-C/eps)
    k_mat = torch.exp(-c / float(eps)).clamp_min(1e-9)

    u = torch.ones((x.shape[0], n), device=x.device, dtype=x.dtype) / n
    v = torch.ones((x.shape[0], m), device=x.device, dtype=x.dtype) / m

    for _ in range(int(n_iters)):
        u = a / (k_mat @ v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
        v = b_mask / (k_mat.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)

    # Transport plan P = diag(u) K diag(v)
    p = u.unsqueeze(-1) * k_mat * v.unsqueeze(1)

    # OT cost = sum P * C over valid pairs.
    return (p * c).sum(dim=(1, 2)).mean()


def gate_entropy_regularizer(gate_probs: torch.Tensor) -> torch.Tensor:
    """
    gate_probs: (B, E), should be a probability distribution.
    Minimizing sum p log p encourages higher entropy (less collapse).
    """
    p = gate_probs.clamp_min(1e-8)
    return (p * torch.log(p)).sum(dim=-1).mean()


def gate_load_balance_regularizer(gate_probs: torch.Tensor) -> torch.Tensor:
    """
    Encourage average routing probability to be close to uniform across experts.
    """
    p = gate_probs.clamp_min(1e-8)
    mean_p = p.mean(dim=0)  # (E,)
    e = mean_p.numel()
    uniform = torch.full_like(mean_p, 1.0 / float(e))
    return F.mse_loss(mean_p, uniform)


def coteach_sample_weight(
    ct_logits: torch.Tensor,
    wsi_logits: torch.Tensor,
    y_true: torch.Tensor,
    mismatch_score: torch.Tensor,
    epoch: int,
    cfg: CoTeachConfig,
    *,
    pos_weight: float,
) -> torch.Tensor:
    """
    Returns per-sample weights for the fusion classification loss.

    Strategy (simplified co-teaching)
    - Compute per-sample BCE losses for CT-only and WSI-only experts.
    - Select small-loss samples (retain rate = 1 - r(epoch)).
    - Cross-update: use CT's selection to weight WSI and vice versa.
    - Apply mismatch-aware reweighting: loss *= (1 + gamma * m).

    This function is optional and only used during training.
    """
    if not cfg.enable:
        return torch.ones_like(y_true, dtype=torch.float32)

    epoch = int(epoch)
    r0 = float(cfg.noise_rate)
    if epoch < int(cfg.warmup_epochs):
        r = r0 * float(epoch + 1) / float(cfg.warmup_epochs)
    else:
        r = r0
    keep = max(1, int(math.ceil((1.0 - r) * y_true.numel())))

    with torch.no_grad():
        m = mismatch_score.view(-1).clamp(0.0, 1.0)
        w_m = (1.0 + float(cfg.mismatch_gamma) * m).detach()

        y = y_true.view(-1).to(dtype=torch.float32)
        w = torch.tensor([float(pos_weight)], dtype=torch.float32, device=y_true.device)

        loss_ct = F.binary_cross_entropy_with_logits(ct_logits.view(-1), y, pos_weight=w, reduction="none") * w_m
        loss_wsi = F.binary_cross_entropy_with_logits(wsi_logits.view(-1), y, pos_weight=w, reduction="none") * w_m

        _, idx_ct = torch.topk(-loss_ct, k=keep, largest=True)   # smallest loss -> largest -loss
        _, idx_wsi = torch.topk(-loss_wsi, k=keep, largest=True)

        w_eff = torch.zeros_like(y, dtype=torch.float32)
        w_eff[idx_ct] += 0.5
        w_eff[idx_wsi] += 0.5
        w_eff = w_eff.clamp(0.0, 1.0)

    return w_eff


class DRGFuseTrainingLoss(nn.Module):
    def __init__(
        self,
        *,
        pos_weight: float,
        beta_low_fpr: float = 0.05,
        weights: LossWeights = LossWeights(),
        coteach: CoTeachConfig = CoTeachConfig(),
        ot_eps: float = 0.05,
        ot_iters: int = 30,
    ) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.beta_low_fpr = float(beta_low_fpr)
        self.weights = weights
        self.coteach = coteach
        self.ot_eps = float(ot_eps)
        self.ot_iters = int(ot_iters)

    def forward(
        self,
        *,
        y_logit: torch.Tensor,
        y_true: torch.Tensor,
        gate_probs: torch.Tensor,
        ct_tokens: torch.Tensor,
        wsi_tokens: torch.Tensor,
        ct_mask: torch.Tensor,
        wsi_mask: torch.Tensor,
        ct_global: torch.Tensor,
        wsi_global: torch.Tensor,
        mismatch_score: torch.Tensor,
        ct_logit: Optional[torch.Tensor] = None,
        wsi_logit: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # Co-teaching weights (optional)
        sample_w = None
        if self.coteach.enable and (ct_logit is not None) and (wsi_logit is not None):
            sample_w = coteach_sample_weight(
                ct_logits=ct_logit,
                wsi_logits=wsi_logit,
                y_true=y_true,
                mismatch_score=mismatch_score,
                epoch=epoch,
                cfg=self.coteach,
                pos_weight=self.pos_weight,
            )

        loss_bce = weighted_bce_with_logits(y_logit, y_true, self.pos_weight, sample_weight=sample_w)

        loss_low_fpr = low_fpr_pairwise_surrogate(y_logit, y_true, beta=self.beta_low_fpr)

        if self.weights.ot != 0.0:
            loss_ot = sinkhorn_ot_distance(
                ct_tokens,
                wsi_tokens,
                ct_mask,
                wsi_mask,
                eps=self.ot_eps,
                n_iters=self.ot_iters,
                cost="cosine",
            )
        else:
            loss_ot = y_logit.new_tensor(0.0)

        if self.weights.mmd != 0.0:
            loss_mmd = mmd_rbf_multi(ct_global, wsi_global)
        else:
            loss_mmd = y_logit.new_tensor(0.0)

        if self.weights.gate_entropy != 0.0:
            loss_gate_ent = gate_entropy_regularizer(gate_probs)
        else:
            loss_gate_ent = y_logit.new_tensor(0.0)

        if self.weights.gate_balance != 0.0:
            loss_gate_bal = gate_load_balance_regularizer(gate_probs)
        else:
            loss_gate_bal = y_logit.new_tensor(0.0)

        total = (
            self.weights.bce * loss_bce
            + self.weights.low_fpr * loss_low_fpr
            + self.weights.ot * loss_ot
            + self.weights.mmd * loss_mmd
            + self.weights.gate_entropy * loss_gate_ent
            + self.weights.gate_balance * loss_gate_bal
        )

        out = {
            "total": total,
            "bce": loss_bce.detach(),
            "low_fpr": loss_low_fpr.detach(),
            "ot": loss_ot.detach(),
            "mmd": loss_mmd.detach(),
            "gate_entropy": loss_gate_ent.detach(),
            "gate_balance": loss_gate_bal.detach(),
        }
        if sample_w is not None:
            out["coteach_weight_mean"] = sample_w.mean().detach()
        return out
