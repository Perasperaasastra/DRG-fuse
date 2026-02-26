#!/usr/bin/env python3
"""
Evaluate a trained checkpoint using a leakage-safe high-specificity protocol.

Protocol
- Compute the Sp95 threshold using *validation negatives only*.
- Apply the same threshold to test/external splits.

Usage (toy):
  python scripts/eval.py --checkpoint outputs/toy_run/run_*/checkpoints/best.pt --config configs/toy.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from drgfuse.data import FeatureNPZDataset, fusion_collate
from drgfuse.metrics import (
    brier_score,
    expected_calibration_error,
    pauc_normalized,
    roc_auc,
    threshold_from_val_negatives,
    tpr_at_specificity,
)
from drgfuse.models.fusion import ControlledInteractionFusion, FusionInputs
from drgfuse.utils import load_yaml, save_json


@torch.no_grad()
def predict_split(model, ds, batch_size: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=fusion_collate)
    model.eval()
    ys, ps, rows = [], [], []
    for batch in loader:
        inp = FusionInputs(
            ct_tokens=batch.ct_tokens.to(device),
            ct_global=batch.ct_global.to(device),
            ct_logit=batch.ct_logit.to(device) if batch.ct_logit is not None else None,
            ct_quality=batch.ct_quality.to(device),
            wsi_tokens=batch.wsi_tokens.to(device),
            wsi_global=batch.wsi_global.to(device),
            wsi_logit=batch.wsi_logit.to(device) if batch.wsi_logit is not None else None,
            wsi_quality=batch.wsi_quality.to(device),
            ct_token_mask=batch.ct_token_mask.to(device),
            wsi_token_mask=batch.wsi_token_mask.to(device),
        )
        out = model(inp)
        y = batch.y_true.cpu().numpy().astype(np.int32)
        p = out.p.detach().cpu().numpy().astype(np.float64)
        ys.append(y)
        ps.append(p)
        for sid, pid, yy, pp in zip(batch.sample_ids, batch.patient_ids, y.tolist(), p.tolist()):
            rows.append({"sample_id": sid, "patient_id": pid, "y_true": yy, "p_pred": pp})
    y_all = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int32)
    p_all = np.concatenate(ps, axis=0) if ps else np.zeros((0,), dtype=np.float64)
    return y_all, p_all, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/toy.yaml")
    ap.add_argument("--out_dir", type=str, default="outputs/eval")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_root = Path(cfg["data"]["root"])
    val_split = str(cfg["data"]["val_split"])
    test_splits = list(cfg["data"]["test_splits"])

    batch_size = int(cfg["train"]["batch_size"])

    ckpt = torch.load(args.checkpoint, map_location="cpu")
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validation split (threshold source)
    ds_val = FeatureNPZDataset(data_root, split=val_split)
    y_val, p_val, df_val = predict_split(model, ds_val, batch_size, device)
    df_val.to_csv(out_dir / f"preds_{val_split}.csv", index=False)

    thr_sp95 = threshold_from_val_negatives(y_val, p_val, specificity=0.95)

    metrics: Dict[str, Dict[str, float]] = {}
    metrics[val_split] = {
        "auc": roc_auc(y_val, p_val),
        "paucn@0.05": pauc_normalized(y_val, p_val, beta=0.05),
        "tpr@sp95": tpr_at_specificity(y_val, p_val, threshold=thr_sp95),
        "threshold_sp95": float(thr_sp95),
        "brier": brier_score(y_val, p_val),
        "ece": expected_calibration_error(y_val, p_val, n_bins=10),
    }

    # Test splits
    for split in test_splits:
        ds = FeatureNPZDataset(data_root, split=split)
        y, p, df = predict_split(model, ds, batch_size, device)
        df.to_csv(out_dir / f"preds_{split}.csv", index=False)
        metrics[split] = {
            "auc": roc_auc(y, p),
            "paucn@0.05": pauc_normalized(y, p, beta=0.05),
            "tpr@sp95": tpr_at_specificity(y, p, threshold=thr_sp95),
            "threshold_sp95": float(thr_sp95),
            "brier": brier_score(y, p),
            "ece": expected_calibration_error(y, p, n_bins=10),
        }

    save_json(out_dir / "metrics.json", metrics)
    print(f"Wrote metrics to {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
