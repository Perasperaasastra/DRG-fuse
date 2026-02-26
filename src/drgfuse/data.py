"""
Dataset utilities for DRGFuse (feature-level fusion).

This repository releases the fusion-side code. Therefore, the primary input format
is *pre-extracted* patient-level features:
- CT token sequence + CT global embedding + CT quality vector
- WSI token sequence + WSI global embedding + WSI quality vector

The default on-disk format is:
data_root/
  metadata.csv
  features/
    <sample_id>.npz

Each per-sample .npz should contain (minimum):
  - ct_tokens: (N_ct, D_ct) float32
  - wsi_tokens: (N_wsi, D_wsi) float32
  - ct_quality: (Q_ct,) float32
  - wsi_quality: (Q_wsi,) float32
Optional keys:
  - ct_global: (D_ct,) float32 (otherwise computed as mean pool of tokens)
  - wsi_global: (D_wsi,) float32 (otherwise computed as mean pool of tokens)
  - ct_logit: () float32 (optional unimodal logit)
  - wsi_logit: () float32 (optional unimodal logit)

metadata.csv columns (minimum):
  - sample_id
  - patient_id
  - label  (binary: 1 = TRG0, 0 = TRG1-3)
  - center (Internal / ExternalA / ExternalB)
  - feature_path (relative to data_root, e.g., features/xxx.npz)
Optional columns:
  - fold / split
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class FusionBatch:
    sample_ids: List[str]
    patient_ids: List[str]
    y_true: torch.Tensor

    ct_global: torch.Tensor
    wsi_global: torch.Tensor
    ct_quality: torch.Tensor
    wsi_quality: torch.Tensor

    ct_tokens: torch.Tensor
    wsi_tokens: torch.Tensor
    ct_token_mask: torch.Tensor
    wsi_token_mask: torch.Tensor

    ct_logit: Optional[torch.Tensor] = None
    wsi_logit: Optional[torch.Tensor] = None


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    return {k: v for k, v in np.load(path, allow_pickle=False).items()}


def _mean_pool(tokens: np.ndarray) -> np.ndarray:
    if tokens.ndim != 2:
        raise ValueError(f"Expected tokens with shape (N,D), got {tokens.shape}")
    if tokens.shape[0] == 0:
        raise ValueError("Token sequence is empty.")
    return tokens.mean(axis=0)


def _pad_2d(seqs: Sequence[np.ndarray], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of 2D arrays (N_i, D) to a single tensor (B, N_max, D),
    returning (padded, mask) where mask is (B, N_max) bool.
    """
    if len(seqs) == 0:
        raise ValueError("Empty sequence list.")
    d = seqs[0].shape[1]
    for x in seqs:
        if x.ndim != 2 or x.shape[1] != d:
            raise ValueError("All token arrays must have shape (N, D) with the same D.")
    n_max = max(x.shape[0] for x in seqs)
    b = len(seqs)
    out = torch.full((b, n_max, d), float(pad_value), dtype=torch.float32)
    mask = torch.zeros((b, n_max), dtype=torch.bool)
    for i, x in enumerate(seqs):
        n = x.shape[0]
        out[i, :n] = torch.from_numpy(x.astype(np.float32))
        mask[i, :n] = True
    return out, mask


class FeatureNPZDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        *,
        centers: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        fold: Optional[int] = None,
        label_col: str = "label",
        center_col: str = "center",
        split_col: str = "split",
        fold_col: str = "fold",
        feature_path_col: str = "feature_path",
        sample_id_col: str = "sample_id",
        patient_id_col: str = "patient_id",
    ) -> None:
        self.data_root = Path(data_root)
        meta_path = self.data_root / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.csv not found under: {self.data_root}")
        self.meta = pd.read_csv(meta_path)

        needed = {label_col, center_col, feature_path_col, sample_id_col, patient_id_col}
        missing = [c for c in needed if c not in self.meta.columns]
        if missing:
            raise ValueError(f"metadata.csv missing columns: {missing}")

        df = self.meta.copy()
        if centers is not None:
            df = df[df[center_col].isin(list(centers))].copy()
        if split is not None:
            if split_col not in df.columns:
                raise ValueError(f"split requested but column '{split_col}' not found in metadata.csv")
            df = df[df[split_col] == split].copy()
        if fold is not None:
            if fold_col not in df.columns:
                raise ValueError(f"fold requested but column '{fold_col}' not found in metadata.csv")
            df = df[df[fold_col] == int(fold)].copy()

        df = df.reset_index(drop=True)
        self.df = df
        self.label_col = label_col
        self.feature_path_col = feature_path_col
        self.sample_id_col = sample_id_col
        self.patient_id_col = patient_id_col

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[int(idx)]
        sample_id = str(row[self.sample_id_col])
        patient_id = str(row[self.patient_id_col])
        y = float(row[self.label_col])

        feat_path = self.data_root / str(row[self.feature_path_col])
        feat = _load_npz(feat_path)

        ct_tokens = feat["ct_tokens"].astype(np.float32)
        wsi_tokens = feat["wsi_tokens"].astype(np.float32)

        ct_quality = feat["ct_quality"].astype(np.float32)
        wsi_quality = feat["wsi_quality"].astype(np.float32)

        ct_global = feat.get("ct_global", _mean_pool(ct_tokens)).astype(np.float32)
        wsi_global = feat.get("wsi_global", _mean_pool(wsi_tokens)).astype(np.float32)

        ct_logit = feat.get("ct_logit", None)
        wsi_logit = feat.get("wsi_logit", None)

        item: Dict[str, object] = {
            "sample_id": sample_id,
            "patient_id": patient_id,
            "y": y,
            "ct_tokens": ct_tokens,
            "wsi_tokens": wsi_tokens,
            "ct_global": ct_global,
            "wsi_global": wsi_global,
            "ct_quality": ct_quality,
            "wsi_quality": wsi_quality,
            "ct_logit": None if ct_logit is None else float(ct_logit),
            "wsi_logit": None if wsi_logit is None else float(wsi_logit),
        }
        return item


def fusion_collate(batch: Sequence[Dict[str, object]]) -> FusionBatch:
    sample_ids = [str(x["sample_id"]) for x in batch]
    patient_ids = [str(x["patient_id"]) for x in batch]
    y = torch.tensor([float(x["y"]) for x in batch], dtype=torch.float32)

    ct_global = torch.from_numpy(np.stack([x["ct_global"] for x in batch]).astype(np.float32))
    wsi_global = torch.from_numpy(np.stack([x["wsi_global"] for x in batch]).astype(np.float32))

    ct_quality = torch.from_numpy(np.stack([x["ct_quality"] for x in batch]).astype(np.float32))
    wsi_quality = torch.from_numpy(np.stack([x["wsi_quality"] for x in batch]).astype(np.float32))

    ct_tokens_list = [x["ct_tokens"] for x in batch]
    wsi_tokens_list = [x["wsi_tokens"] for x in batch]
    ct_tokens, ct_mask = _pad_2d(ct_tokens_list)
    wsi_tokens, wsi_mask = _pad_2d(wsi_tokens_list)

    ct_logit_vals = [x.get("ct_logit", None) for x in batch]
    wsi_logit_vals = [x.get("wsi_logit", None) for x in batch]
    ct_logit = None if any(v is None for v in ct_logit_vals) else torch.tensor(ct_logit_vals, dtype=torch.float32)
    wsi_logit = None if any(v is None for v in wsi_logit_vals) else torch.tensor(wsi_logit_vals, dtype=torch.float32)

    return FusionBatch(
        sample_ids=sample_ids,
        patient_ids=patient_ids,
        y_true=y,
        ct_global=ct_global,
        wsi_global=wsi_global,
        ct_quality=ct_quality,
        wsi_quality=wsi_quality,
        ct_tokens=ct_tokens,
        wsi_tokens=wsi_tokens,
        ct_token_mask=ct_mask,
        wsi_token_mask=wsi_mask,
        ct_logit=ct_logit,
        wsi_logit=wsi_logit,
    )
