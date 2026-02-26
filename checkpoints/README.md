This directory stores model checkpoints.

- `toy_model.pt` is a small checkpoint trained on the synthetic toy dataset under `data/example/`.
  It is provided only to demonstrate that the inference pipeline runs end-to-end.
  It is **not** a model trained on the real study cohort and should not be used to claim any
  performance numbers.

For real experiments, please save checkpoints under `outputs/` (ignored by `.gitignore`).
