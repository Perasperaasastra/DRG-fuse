# Data description

## Task

Binary classification:

- Positive (label=1): **TRG0**
- Negative (label=0): **TRG1–3**

The clinical endpoint is **TRG0 vs TRG1–3**, focusing on high-specificity deployment settings.

## Cohort (private)

The cohort is a **private multi-center clinical dataset**:

- **Internal**: n=525  
- **ExternalA**: n=51  
- **ExternalB**: n=62  

Each case contains:
- a traceable TRG label,
- pre-treatment CT,
- pre-treatment biopsy WSI (whole-slide image).

Records with missing/invalid TRG are excluded.
CT and WSI are paired at the patient level using in-hospital identifiers.

**Data release:** due to privacy constraints, the real cohort is **not publicly released**.

## Split principle (leakage-safe)

Patient-level, stratified **5-fold cross-validation** is performed on **Internal**.

For each fold:
- Internal is split into **train / val / test**.
- **Val** is used for:
  - early stopping / model selection,
  - determining the **Sp95 threshold** using **validation negatives only**.
- The threshold is then **fixed** and applied to:
  - the corresponding Internal test split,
  - ExternalA and ExternalB (held out; never used in training or threshold selection).

External centers do **not** participate in any training or model/threshold selection.

## Released toy example (synthetic)

This repository includes a **synthetic toy dataset** under `data/example/` to validate that
the pipeline runs end-to-end:

- `data/example/metadata.csv`
- `data/example/features/*.npz`

No real patient data are included.

## Feature format (what the fusion code expects)

This repository operates at the **feature level**.
You should first extract per-patient representations from CT/WSI and save them as `.npz`.

Directory layout:

```text
<data_root>/
  metadata.csv
  features/
    <sample_id>.npz
```

`metadata.csv` required columns:
- `sample_id`
- `patient_id`
- `label` (0/1)
- `center` (Internal/ExternalA/ExternalB)
- `split` (train/val/test_internal/test_externalA/test_externalB, or your own names)
- `feature_path` (relative path to the `.npz` file)

Each `<sample_id>.npz` must include:
- `ct_tokens`: (N_ct, D_ct) float32
- `wsi_tokens`: (N_wsi, D_wsi) float32
- `ct_quality`: (Q_ct,) float32
- `wsi_quality`: (Q_wsi,) float32

Optional keys:
- `ct_global`: (D_ct,) float32 (otherwise mean-pool of `ct_tokens`)
- `wsi_global`: (D_wsi,) float32 (otherwise mean-pool of `wsi_tokens`)
- `ct_logit`, `wsi_logit`: optional unimodal logits (scalars)

See `src/drgfuse/data.py` for the exact loader.
