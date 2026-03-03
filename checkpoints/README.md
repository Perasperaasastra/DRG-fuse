This directory is intentionally kept **empty** in the anonymized snapshot.

- Do **not** commit real study checkpoints (weights) to the anonymous repository.
- For the toy demo, you can generate a small checkpoint locally by running:
  - `python scripts/make_toy_data.py --out_root data/example --seed 0`
  - `python scripts/train.py --config configs/toy.yaml`

The best checkpoint will be written under `outputs/` (ignored by `.gitignore`).
