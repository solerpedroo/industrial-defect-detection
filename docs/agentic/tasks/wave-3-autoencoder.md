# Task: `wave-3-autoencoder`

- **Status:** proposed
- **Wave:** 3 — Convolutional autoencoder
- **Owner / role:** unassigned
- **Model or category IDs:** `autoencoder`
- **Depends on:** Wave 2, ADR that adds PyTorch
- **Allowed files:** `ml/models/autoencoder/`, `ml/training/`, `scripts/train_autoencoder.py`,
  autoencoder tests, model catalog status.

## Goal

Train a convolutional autoencoder on normal images with Adam, MSE, early
stopping on the normal validation split, and a saved checkpoint. Inference
returns a reconstruction and an error map.

## Contract and conventions

Input tensors are documented (size, normalization). The error map is a
non-negative float array. A high reconstruction error is not automatically a
defect label until Wave 5 supplies a threshold.

## Plan and acceptance criteria

- [ ] Encoder and decoder restore spatial size.
- [ ] Training CLI writes a checkpoint outside Git and a loss history JSON.
- [ ] Synthetic overfitting test: loss falls on a constant image batch.
- [ ] Catalog status stays `planned` until the API can load the checkpoint.

## Tests and commands

```bash
.venv/bin/pytest backend/tests -q
```

## Non-goals

No PatchCore, no pixel AUROC, no UI heatmap yet.

## Evidence / handoff
