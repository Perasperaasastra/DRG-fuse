#!/usr/bin/env python3
"""
Template: extract CT-side features for DRGFuse.

This repository focuses on fusion at the feature level. This script is provided as a
*template* to show how to create per-patient feature files in the expected format.

You should implement/plug in your own CT pipeline components, e.g.:
- segmentation model (for habitat masks)
- CT encoder / foundation model (for feature maps)
- pooling strategy (for habitat tokens)
- CT unimodal head (optional, produces ct_logit)

Output per patient:
  ct_tokens, ct_quality, (optional ct_global, ct_logit)

See DATA.md for the full `.npz` schema.

NOTE: This template does not ship any clinical data or pretrained weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


def extract_ct_features(ct_path: Path) -> Dict[str, np.ndarray]:
    """
    Implement your CT feature extraction here.

    Expected outputs:
      - ct_tokens: (N_ct, D_ct)
      - ct_quality: (Q_ct,)
    Optional:
      - ct_global: (D_ct,)
      - ct_logit: scalar
    """
    raise NotImplementedError("Implement CT feature extraction for your dataset.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct_path", type=str, required=True, help="Path to a single CT volume.")
    ap.add_argument("--out_npz", type=str, required=True, help="Output .npz path.")
    args = ap.parse_args()

    ct_path = Path(args.ct_path)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    feat = extract_ct_features(ct_path)

    # Validate minimal keys
    for k in ["ct_tokens", "ct_quality"]:
        if k not in feat:
            raise ValueError(f"Missing key: {k}")

    np.savez_compressed(out_npz, **feat)
    print(f"Wrote CT features to: {out_npz}")


if __name__ == "__main__":
    main()
