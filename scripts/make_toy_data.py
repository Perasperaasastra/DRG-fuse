#!/usr/bin/env python3
"""
Create a synthetic toy dataset in the expected feature format.

This is intended for:
- Verifying that training/evaluation/inference scripts run end-to-end.
- Providing a small example that can be shipped with the anonymized repository.

No real clinical data are included.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _make_sample_features(
    rng: np.random.Generator,
    *,
    label: int,
    d_ct: int = 64,
    d_wsi: int = 128,
    q_ct: int = 8,
    q_wsi: int = 6,
) -> dict:
    n_ct = int(rng.integers(4, 9))
    n_wsi = int(rng.integers(32, 65))

    ct_tokens = rng.normal(0.0, 1.0, size=(n_ct, d_ct)).astype(np.float32)
    wsi_tokens = rng.normal(0.0, 1.0, size=(n_wsi, d_wsi)).astype(np.float32)

    # Inject a weak signal correlated with the label.
    if int(label) == 1:
        ct_tokens[:, 0] += 0.8
        wsi_tokens[:, 0] += 0.6
    else:
        ct_tokens[:, 0] -= 0.2
        wsi_tokens[:, 0] -= 0.1

    ct_quality = rng.uniform(0.0, 1.0, size=(q_ct,)).astype(np.float32)
    wsi_quality = rng.uniform(0.0, 1.0, size=(q_wsi,)).astype(np.float32)

    ct_global = ct_tokens.mean(axis=0).astype(np.float32)
    wsi_global = wsi_tokens.mean(axis=0).astype(np.float32)

    return {
        "ct_tokens": ct_tokens,
        "wsi_tokens": wsi_tokens,
        "ct_quality": ct_quality,
        "wsi_quality": wsi_quality,
        "ct_global": ct_global,
        "wsi_global": wsi_global,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="data/example")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_internal", type=int, default=60)
    ap.add_argument("--n_externalA", type=int, default=20)
    ap.add_argument("--n_externalB", type=int, default=20)
    ap.add_argument("--pos_rate", type=float, default=0.25)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    feat_dir = out_root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    rng = _rng(args.seed)

    rows = []
    sample_counter = 0

    def add_samples(center: str, n: int) -> None:
        nonlocal sample_counter
        for _ in range(int(n)):
            sample_counter += 1
            sample_id = f"S{sample_counter:04d}"
            patient_id = f"P{sample_counter:04d}"
            label = int(rng.random() < float(args.pos_rate))

            feat = _make_sample_features(rng, label=label)

            rel_path = Path("features") / f"{sample_id}.npz"
            np.savez_compressed(out_root / rel_path, **feat)

            rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "label": label,
                    "center": center,
                    "feature_path": str(rel_path).replace("\\", "/"),
                }
            )

    add_samples("Internal", args.n_internal)
    add_samples("ExternalA", args.n_externalA)
    add_samples("ExternalB", args.n_externalB)

    meta = pd.DataFrame(rows)
    # Make deterministic splits.
    internal = meta[meta["center"] == "Internal"].copy()
    idx = np.arange(len(internal))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    internal.loc[internal.index[train_idx], "split"] = "train"
    internal.loc[internal.index[val_idx], "split"] = "val"
    internal.loc[internal.index[test_idx], "split"] = "test_internal"

    meta.loc[meta["center"] == "Internal", "split"] = internal["split"].values
    meta.loc[meta["center"] == "ExternalA", "split"] = "test_externalA"
    meta.loc[meta["center"] == "ExternalB", "split"] = "test_externalB"

    (out_root / "metadata.csv").write_text(meta.to_csv(index=False), encoding="utf-8")

    print(f"Wrote metadata.csv with {len(meta)} samples to: {out_root}")
    print("Split counts:")
    print(meta["split"].value_counts())


if __name__ == "__main__":
    main()
