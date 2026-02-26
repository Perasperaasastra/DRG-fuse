#!/usr/bin/env python3
"""
Train the fusion-side model on pre-extracted features.

Expected data format:
- See DATA.md and src/drgfuse/data.py

Typical usage (toy dataset):
  python scripts/make_toy_data.py --out_root data/example
  python scripts/train.py --config configs/toy.yaml

Outputs:
  <output.dir>/
    run_<timestamp>/
      config_snapshot.json
      checkpoints/
        best.pt
        last.pt
      val_predictions.csv
      metrics_val.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from drgfuse.data import FeatureNPZDataset, fusion_collate
from drgfuse.losses import CoTeachConfig, DRGFuseTrainingLoss, LossWeights
from drgfuse.metrics import (
    brier_score,
    expected_calibration_error,
    pauc_normalized,
    roc_auc,
    threshold_from_val_negatives,
    tpr_at_specificity,
)
from drgfuse.models.fusion import ControlledInteractionFusion, FusionInputs
from drgfuse.utils import load_yaml, save_json, set_seed


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _compute_pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos <= 0:
        raise ValueError("No positive samples in train split.")
    return float(n_neg) / float(n_pos)


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval()
    ys = []
    ps = []
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
        y = batch.y_true.cpu().numpy().astype(np.int32)
        p = out.p.detach().cpu().numpy().astype(np.float64)
        ys.append(y)
        ps.append(p)
        for sid, pid, yy, pp in zip(batch.sample_ids, batch.patient_ids, y.tolist(), p.tolist()):
            rows.append({"sample_id": sid, "patient_id": pid, "y_true": yy, "p_pred": pp})
    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(ps, axis=0)
    df = pd.DataFrame(rows)
    return y_all, p_all, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/toy.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    out_root = Path(cfg["output"]["dir"])
    run_dir = out_root / f"run_{_timestamp()}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config_snapshot.json", cfg)

    data_root = Path(cfg["data"]["root"])
    train_split = str(cfg["data"]["train_split"])
    val_split = str(cfg["data"]["val_split"])

    ds_train = FeatureNPZDataset(data_root, split=train_split)
    ds_val = FeatureNPZDataset(data_root, split=val_split)

    dl_train = DataLoader(
        ds_train,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=fusion_collate,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=fusion_collate,
        pin_memory=True,
    )

    y_train = ds_train.df["label"].to_numpy().astype(int)
    pos_weight = cfg["train"].get("pos_weight", None)
    if pos_weight is None:
        pos_weight = _compute_pos_weight(y_train)
    pos_weight = float(pos_weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    ).to(device)

    lcfg = cfg["loss"]
    wcfg = lcfg.get("weights", {})
    weights = LossWeights(
        bce=float(wcfg.get("bce", 1.0)),
        low_fpr=float(wcfg.get("low_fpr", 1.0)),
        ot=float(wcfg.get("ot", 0.0)),
        mmd=float(wcfg.get("mmd", 0.0)),
        gate_entropy=float(wcfg.get("gate_entropy", 0.0)),
        gate_balance=float(wcfg.get("gate_balance", 0.0)),
    )
    ccfg = lcfg.get("coteach", {})
    coteach = CoTeachConfig(
        enable=bool(ccfg.get("enable", False)),
        noise_rate=float(ccfg.get("noise_rate", 0.2)),
        warmup_epochs=int(ccfg.get("warmup_epochs", 5)),
        mismatch_gamma=float(ccfg.get("mismatch_gamma", 1.0)),
    )

    loss_fn = DRGFuseTrainingLoss(
        pos_weight=pos_weight,
        beta_low_fpr=float(lcfg.get("beta_low_fpr", 0.05)),
        weights=weights,
        coteach=coteach,
        ot_eps=float(lcfg.get("ot_eps", 0.05)),
        ot_iters=int(lcfg.get("ot_iters", 30)),
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_key = -1.0
    best_path = ckpt_dir / "best.pt"

    epochs = int(cfg["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"epoch {epoch}/{epochs}", ncols=100)
        total_loss = 0.0
        n_seen = 0

        for batch in pbar:
            opt.zero_grad(set_to_none=True)

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

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(inp)
                loss_out = loss_fn(
                    y_logit=out.y,
                    y_true=batch.y_true.to(device),
                    gate_probs=out.gate_weights,
                    ct_tokens=inp.ct_tokens,
                    wsi_tokens=inp.wsi_tokens,
                    ct_mask=inp.ct_token_mask,
                    wsi_mask=inp.wsi_token_mask,
                    ct_global=inp.ct_global,
                    wsi_global=inp.wsi_global,
                    mismatch_score=out.mismatch_score,
                    ct_logit=inp.ct_logit,
                    wsi_logit=inp.wsi_logit,
                    epoch=epoch - 1,
                )
                loss = loss_out["total"]

            scaler.scale(loss).backward()

            grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(opt)
            scaler.update()

            bs = int(batch.y_true.numel())
            total_loss += float(loss.detach().cpu().item()) * bs
            n_seen += bs
            pbar.set_postfix({"loss": total_loss / max(1, n_seen)})

        # Validation
        y_val, p_val, df_val = _predict(model, dl_val, device)
        auc = roc_auc(y_val, p_val)
        paucn = pauc_normalized(y_val, p_val, beta=0.05)
        thr = threshold_from_val_negatives(y_val, p_val, specificity=0.95)
        tpr = tpr_at_specificity(y_val, p_val, threshold=thr)
        br = brier_score(y_val, p_val)
        ece = expected_calibration_error(y_val, p_val, n_bins=10)

        metrics = {
            "epoch": epoch,
            "val": {
                "auc": auc,
                "paucn@0.05": paucn,
                "tpr@sp95": tpr,
                "threshold_sp95": thr,
                "brier": br,
                "ece": ece,
            },
        }
        save_json(run_dir / "metrics_val.json", metrics)
        df_val.to_csv(run_dir / "val_predictions.csv", index=False)

        # Model selection: use val pAUCn (low-FPR focused).
        if np.isfinite(paucn) and paucn > best_key:
            best_key = float(paucn)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "pos_weight": pos_weight,
                },
                best_path,
            )

        # Always save last
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": cfg,
                "pos_weight": pos_weight,
            },
            ckpt_dir / "last.pt",
        )

        print(f"[val] epoch={epoch} auc={auc:.4f} paucn@0.05={paucn:.4f} tpr@sp95={tpr:.4f}")

    print(f"Training finished. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
