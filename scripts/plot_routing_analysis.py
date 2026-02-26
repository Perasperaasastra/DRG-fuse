#!/usr/bin/env python3
"""
Generate toy visualizations similar to the paper's routing analysis figure.

This script is optional and intended to:
- Demonstrate how to compute mismatch and routing weights.
- Provide a template for reproducing routing analysis plots on real data.

Usage (toy):
  python scripts/plot_routing_analysis.py \
    --checkpoint outputs/toy_run/run_*/checkpoints/best.pt \
    --config configs/toy.yaml \
    --out_dir outputs/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from drgfuse.data import FeatureNPZDataset, fusion_collate
from drgfuse.metrics import threshold_from_val_negatives, tpr_at_specificity
from drgfuse.models.fusion import ControlledInteractionFusion, FusionInputs
from drgfuse.utils import load_yaml


@torch.no_grad()
def collect(model, ds, batch_size: int, device: torch.device) -> pd.DataFrame:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=fusion_collate)
    rows = []
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

        # Unimodal logits from internal heads
        y_ct = model.ct_unimodal_head(inp.ct_global).squeeze(-1)
        y_wsi = model.wsi_unimodal_head(inp.wsi_global).squeeze(-1)

        for i in range(len(batch.sample_ids)):
            rows.append(
                {
                    "sample_id": batch.sample_ids[i],
                    "patient_id": batch.patient_ids[i],
                    "y_true": float(batch.y_true[i].item()),
                    "p_full": float(out.p[i].item()),
                    "p_ct": float(torch.sigmoid(y_ct[i]).item()),
                    "p_wsi": float(torch.sigmoid(y_wsi[i]).item()),
                    "mismatch": float(out.mismatch_score[i].item()),
                    "w_ct": float(out.gate_weights[i, 0].item()),
                    "w_wsi": float(out.gate_weights[i, 1].item()),
                    "w_fuse": float(out.gate_weights[i, 2].item()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/toy.yaml")
    ap.add_argument("--out_dir", type=str, default="outputs/plots")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_root = Path(cfg["data"]["root"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    model.eval()

    batch_size = int(cfg["train"]["batch_size"])

    meta = pd.read_csv(data_root / "metadata.csv")

    # Collect val split for thresholds
    val_split = str(cfg["data"]["val_split"])
    ds_val = FeatureNPZDataset(data_root, split=val_split)
    df_val = collect(model, ds_val, batch_size, device)
    df_val = df_val.merge(meta[["sample_id", "center", "split"]], on="sample_id", how="left")

    thr_full = threshold_from_val_negatives(df_val["y_true"].values, df_val["p_full"].values, specificity=0.95)
    thr_ct = threshold_from_val_negatives(df_val["y_true"].values, df_val["p_ct"].values, specificity=0.95)
    thr_wsi = threshold_from_val_negatives(df_val["y_true"].values, df_val["p_wsi"].values, specificity=0.95)

    # Collect test splits
    test_splits = list(cfg["data"]["test_splits"])
    dfs = []
    for sp in test_splits:
        ds = FeatureNPZDataset(data_root, split=sp)
        df = collect(model, ds, batch_size, device)
        df = df.merge(meta[["sample_id", "center", "split"]], on="sample_id", how="left")
        dfs.append(df)
    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    # (a) TPR@Sp95 across centers for CT-only / WSI-only / Full
    centers = ["Internal", "ExternalA", "ExternalB"]
    tpr_full = []
    tpr_ct = []
    tpr_wsi = []
    for c in centers:
        sub = df_all[df_all["center"] == c]
        tpr_full.append(tpr_at_specificity(sub["y_true"].values, sub["p_full"].values, thr_full))
        tpr_ct.append(tpr_at_specificity(sub["y_true"].values, sub["p_ct"].values, thr_ct))
        tpr_wsi.append(tpr_at_specificity(sub["y_true"].values, sub["p_wsi"].values, thr_wsi))

    plt.figure()
    plt.plot(centers, tpr_ct, marker="o", label="CT-only")
    plt.plot(centers, tpr_wsi, marker="s", label="WSI-only")
    plt.plot(centers, tpr_full, marker="D", label="Full fusion")
    plt.ylabel("TPR @ Spec=0.95")
    plt.title("Sensitivity at Sp95 (toy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "toy_tpr_sp95.png", dpi=200)
    plt.close()

    # (b) mismatch vs fusion routing weight
    plt.figure()
    plt.scatter(df_all["mismatch"].values, df_all["w_fuse"].values, s=12, alpha=0.6)
    plt.xlabel("Mismatch score m")
    plt.ylabel("Fusion expert weight w_fuse")
    plt.title("Mismatch vs fusion routing weight (toy)")
    plt.tight_layout()
    plt.savefig(out_dir / "toy_mismatch_vs_wfuse.png", dpi=200)
    plt.close()

    # (c) dominant expert among predicted positives at Sp95 (using full model threshold)
    df_all = df_all.copy()
    df_all["pred_pos"] = df_all["p_full"].values > thr_full
    df_pos = df_all[df_all["pred_pos"]].copy()
    df_pos["dominant"] = df_pos[["w_ct", "w_wsi", "w_fuse"]].values.argmax(axis=1)

    # 0:CT, 1:WSI, 2:Fuse
    props = np.zeros((len(centers), 3), dtype=np.float64)
    for i, c in enumerate(centers):
        sub = df_pos[df_pos["center"] == c]
        if len(sub) == 0:
            continue
        counts = np.bincount(sub["dominant"].values.astype(int), minlength=3)
        props[i] = counts / counts.sum()

    plt.figure()
    bottom = np.zeros(len(centers))
    labels = ["CT", "WSI", "Fuse"]
    for j in range(3):
        plt.bar(centers, props[:, j], bottom=bottom, label=labels[j])
        bottom += props[:, j]
    plt.ylabel("Proportion among predicted TRG0")
    plt.title("Dominant expert at Sp95 (toy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "toy_dominant_expert.png", dpi=200)
    plt.close()

    print(f"Wrote toy plots to: {out_dir}")


if __name__ == "__main__":
    main()
