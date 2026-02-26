"""
Evaluation utilities for patient-level binary classification.

Task definition
- Binary target: 1 = TRG0, 0 = TRG1-3
- High-specificity operating points are used (e.g., Sp=0.95).

Protocol for thresholds (leakage-safe)
1) Compute thresholds using *validation negatives only*.
2) Apply those thresholds unchanged to test/external splits.
3) Predict positive if score > threshold.

Provided metrics
- AUC (ROC AUC)
- pAUCn@beta: normalized partial AUC over FPR in [0, beta]
- TPR@Sp95: sensitivity at specificity 0.95
- Brier score and ECE (optional)

This file is designed to be used by scripts/eval.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception as e:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for evaluation (roc_auc_score, roc_curve).") from e


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def roc_auc(y_true, y_score) -> float:
    y_true = _to_numpy(y_true).astype(np.int32)
    y_score = _to_numpy(y_score).astype(np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def pauc_normalized(y_true, y_score, beta: float = 0.05) -> float:
    """
    Normalized partial AUC over FPR in [0, beta], scaled to [0, 1].

    Implementation:
    - Compute ROC curve (fpr, tpr)
    - Trapezoidal integration of tpr over fpr within [0, beta]
    - Divide by beta
    """
    y_true = _to_numpy(y_true).astype(np.int32)
    y_score = _to_numpy(y_score).astype(np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    beta = float(beta)
    if beta <= 0:
        raise ValueError("beta must be > 0")
    if fpr[0] > 0:
        fpr = np.concatenate([[0.0], fpr])
        tpr = np.concatenate([[0.0], tpr])

    if beta >= 1.0:
        area = np.trapz(tpr, fpr)
        return float(area)

    # Clip to [0, beta]
    if fpr[-1] < beta:
        fpr_c = np.concatenate([fpr, [beta]])
        tpr_c = np.concatenate([tpr, [tpr[-1]]])
    else:
        idx = np.searchsorted(fpr, beta, side="right")
        fpr_c = np.concatenate([fpr[:idx], [beta]])
        tpr_at_beta = np.interp(beta, fpr, tpr)
        tpr_c = np.concatenate([tpr[:idx], [tpr_at_beta]])

    area = np.trapz(tpr_c, fpr_c)
    return float(area / beta)


def threshold_from_val_negatives(y_val_true, y_val_score, specificity: float = 0.95) -> float:
    """
    Compute score threshold using validation negatives only.

    Specificity definition on negatives:
      specificity = P(score <= threshold | negative)

    Therefore threshold is the specificity-quantile of negative scores.
    """
    y_val_true = _to_numpy(y_val_true).astype(np.int32)
    y_val_score = _to_numpy(y_val_score).astype(np.float64)
    neg_scores = y_val_score[y_val_true == 0]
    if neg_scores.size == 0:
        raise ValueError("No negative samples in validation set.")
    s = float(specificity)
    if not (0 < s < 1):
        raise ValueError("specificity must be in (0, 1)")
    return float(np.quantile(neg_scores, s))


def tpr_at_specificity(y_true, y_score, threshold: float) -> float:
    y_true = _to_numpy(y_true).astype(np.int32)
    y_score = _to_numpy(y_score).astype(np.float64)
    pred_pos = y_score > float(threshold)
    pos = (y_true == 1)
    if pos.sum() == 0:
        return float("nan")
    return float((pred_pos & pos).sum() / pos.sum())


def brier_score(y_true, y_prob) -> float:
    y_true = _to_numpy(y_true).astype(np.float64)
    y_prob = _to_numpy(y_prob).astype(np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Simple ECE with uniform bins in [0, 1].
    """
    y_true = _to_numpy(y_true).astype(np.float64)
    y_prob = _to_numpy(y_prob).astype(np.float64)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < len(bins) - 2 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


@dataclass(frozen=True)
class EvalSummary:
    auc: float
    paucn_005: float
    tpr_sp95: float
    threshold_sp95: float
    brier: float
    ece: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "auc": float(self.auc),
            "paucn@0.05": float(self.paucn_005),
            "tpr@sp95": float(self.tpr_sp95),
            "threshold_sp95": float(self.threshold_sp95),
            "brier": float(self.brier),
            "ece": float(self.ece),
        }
