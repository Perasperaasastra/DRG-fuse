#!/usr/bin/env python3
"""
Run inference for a single sample.

Example:
  python scripts/infer.py \
    --checkpoint checkpoints/toy_model.pt \
    --data_root data/example \
    --sample_id S0001
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from drgfuse.models.fusion import ControlledInteractionFusion, FusionInputs


def _load_feature(data_root: Path, sample_id: str) -> Dict[str, np.ndarray]:
    meta = pd.read_csv(data_root / "metadata.csv")
    row = meta[meta["sample_id"] == sample_id]
    if len(row) != 1:
        raise ValueError(f"sample_id not found or not unique: {sample_id}")
    feat_path = data_root / row.iloc[0]["feature_path"]
    feat = {k: v for k, v in np.load(feat_path, allow_pickle=False).items()}
    return feat, row.iloc[0].to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--sample_id", type=str, required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("Checkpoint does not contain config.")

    mcfg = cfg["model"]
    model = ControlledInteractionFusion(
        d_ct_token=int(mcfg["d_ct_token"]),
        d_wsi_token=int(mcfg["d_wsi_token"]),
        d_ct_global=int(mcfg["d_ct_global"]),
        d_wsi_global=int(mcfg["d_wsi_global"]),
        q_ct_dim=int(mcfg["q_ct_dim"]),
        q_wsi_dim=int(mcfg["q_wsi_dim"]),
        d_model=int(mcfg["d_model"]),
        d_delta=int(mcfg["d_delta"]),
        n_layers=int(mcfg["n_layers"]),
        n_heads=int(mcfg["n_heads"]),
        dropout=float(mcfg["dropout"]),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    data_root = Path(args.data_root)
    feat, meta_row = _load_feature(data_root, args.sample_id)

    ct_tokens = torch.from_numpy(feat["ct_tokens"].astype(np.float32)).unsqueeze(0).to(device)
    wsi_tokens = torch.from_numpy(feat["wsi_tokens"].astype(np.float32)).unsqueeze(0).to(device)

    ct_quality = torch.from_numpy(feat["ct_quality"].astype(np.float32)).unsqueeze(0).to(device)
    wsi_quality = torch.from_numpy(feat["wsi_quality"].astype(np.float32)).unsqueeze(0).to(device)

    ct_global = torch.from_numpy(feat.get("ct_global", feat["ct_tokens"].mean(axis=0)).astype(np.float32)).unsqueeze(0).to(device)
    wsi_global = torch.from_numpy(feat.get("wsi_global", feat["wsi_tokens"].mean(axis=0)).astype(np.float32)).unsqueeze(0).to(device)

    inp = FusionInputs(
        ct_tokens=ct_tokens,
        ct_global=ct_global,
        ct_logit=None,
        ct_quality=ct_quality,
        wsi_tokens=wsi_tokens,
        wsi_global=wsi_global,
        wsi_logit=None,
        wsi_quality=wsi_quality,
        ct_token_mask=torch.ones((1, ct_tokens.shape[1]), dtype=torch.bool, device=device),
        wsi_token_mask=torch.ones((1, wsi_tokens.shape[1]), dtype=torch.bool, device=device),
    )

    with torch.no_grad():
        out = model(inp)

    print("Sample metadata:")
    for k in ["sample_id", "patient_id", "center", "split", "label"]:
        if k in meta_row:
            print(f"  {k}: {meta_row[k]}")

    print("\nPrediction:")
    print(f"  p(TRG0) = {out.p.item():.6f}")
    print(f"  logit   = {out.y.item():.6f}")

    w = out.gate_weights.squeeze(0).detach().cpu().numpy()
    print("\nPost-gate routing weights [CT, WSI, Fuse]:")
    print(f"  {w.tolist()}")

    print("\nMismatch score:")
    print(f"  m = {out.mismatch_score.item():.6f} (higher means more conflict)")


if __name__ == "__main__":
    main()
