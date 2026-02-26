#!/usr/bin/env python3
"""
Template: extract WSI-side features for DRGFuse.

This repository focuses on fusion at the feature level. This script is provided as a
*template* to show how to create per-patient feature files in the expected format.

You should implement/plug in your own WSI pipeline components, e.g.:
- tissue detection + tiling
- patch encoder (e.g., a foundation model)
- slide-level aggregator (optional, produces wsi_global and/or wsi_logit)
- quality vector computation

Output per patient:
  wsi_tokens, wsi_quality, (optional wsi_global, wsi_logit)

See DATA.md for the full `.npz` schema.

NOTE: This template does not ship any clinical data or pretrained weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


def extract_wsi_features(wsi_path: Path) -> Dict[str, np.ndarray]:
    """
    Implement your WSI feature extraction here.

    Expected outputs:
      - wsi_tokens: (N_wsi, D_wsi)
      - wsi_quality: (Q_wsi,)
    Optional:
      - wsi_global: (D_wsi,)
      - wsi_logit: scalar
    """
    raise NotImplementedError("Implement WSI feature extraction for your dataset.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_path", type=str, required=True, help="Path to a single WSI file.")
    ap.add_argument("--out_npz", type=str, required=True, help="Output .npz path.")
    args = ap.parse_args()

    wsi_path = Path(args.wsi_path)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    feat = extract_wsi_features(wsi_path)

    # Validate minimal keys
    for k in ["wsi_tokens", "wsi_quality"]:
        if k not in feat:
            raise ValueError(f"Missing key: {k}")

    np.savez_compressed(out_npz, **feat)
    print(f"Wrote WSI features to: {out_npz}")


if __name__ == "__main__":
    main()
