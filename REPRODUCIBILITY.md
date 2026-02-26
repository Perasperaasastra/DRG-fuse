# Reproducibility guide

This repository releases **fusion-side code** and a **toy demo**.
Reproducing the manuscript results requires access to the private cohort and
pre-extracted CT/WSI features.

---

## 1) Environment setup

Option A (conda):

```bash
conda env create -f environment.yml
conda activate drgfuse
pip install -e .
```

Option B (pip):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 2) Toy demo (runs without any real data)

Generate synthetic features:

```bash
python scripts/make_toy_data.py --out_root data/example --seed 0
```

Train:

```bash
python scripts/train.py --config configs/toy.yaml
```

Evaluate:

```bash
python scripts/eval.py --config configs/toy.yaml --checkpoint <PATH_TO_BEST_PT>
```

Run inference on one sample:

```bash
python scripts/infer.py --checkpoint checkpoints/toy_model.pt --data_root data/example --sample_id S0001
```

Optional: generate toy routing-analysis plots:

```bash
python scripts/plot_routing_analysis.py --checkpoint <PATH_TO_BEST_PT> --config configs/toy.yaml --out_dir outputs/plots
```

---

## 3) Running on a private dataset (feature-level)

### 3.1 Prepare extracted features

Follow `DATA.md` to create:

- `<data_root>/metadata.csv`
- `<data_root>/features/<sample_id>.npz`

Then create a config, e.g.:

- start from `configs/template.yaml`
- set `data.root` to your feature directory (use a relative path in your own repo)
- set feature dimensions (`d_ct_token`, `d_wsi_token`, etc.)

### 3.2 Training protocol

Recommended settings (as used in the manuscript):

- Optimizer: **AdamW**
- Epochs: up to **50**
- Batch size: **4–8**
- Classification loss: **weighted BCE** (pos_weight computed on train split)
- Low-FPR objective (training-only): combine BCE with a **pAUC surrogate** focused on **FPR ∈ [0, 0.05]**
- Alignment guards (training-only): **OT + MMD**
- Validation:
  - early stopping and model selection
  - compute the **Sp95 threshold** using **validation negatives only**

Run:

```bash
python scripts/train.py --config configs/your_config.yaml
```

### 3.3 Evaluation protocol

Leakage-safe thresholding:

1) Compute the threshold on **val negatives only** (Sp95).
2) Fix the threshold and evaluate on test/external splits.

Run:

```bash
python scripts/eval.py --config configs/your_config.yaml --checkpoint <BEST_CHECKPOINT_PT>
```

---

## 4) Results assets included in this snapshot

Manuscript tables:

- `assets/tables/table1.tex`
- `assets/tables/table2.tex`
- `assets/tables/table3.tex`

Supplementary calibration tables:

- `assets/tables/supplementary/*.md`

Figures:

- `assets/figures/fig1_overview.png`
- `assets/figures/fig2_routing.png`

---

## 5) Final anonymization checklist

Before pushing to any remote repository, run:

```bash
python scripts/sanitize_scan.py --root . --fail_on_findings
```

Also ensure:
- no `.git` history from previous projects is included (start from a fresh repo),
- git commits use anonymous author name/email,
- no raw data, logs, caches, or internal paths are committed.
