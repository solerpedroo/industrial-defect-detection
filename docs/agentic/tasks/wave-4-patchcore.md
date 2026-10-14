# Task: `wave-4-patchcore`

- **Status:** proposed
- **Wave:** 4 — PatchCore integration
- **Owner / role:** unassigned
- **Model or category IDs:** `patchcore`
- **Depends on:** Wave 1, ADR that adds Anomalib
- **Allowed files:** `ml/models/patchcore/`, `backend/app/inference/`, PatchCore tests.

## Goal

Wrap Anomalib PatchCore in a project adapter. Build a normal-patch memory with
ResNet-18 and return an anomaly score plus a spatial map. Expose the same
prediction dataclass as the autoencoder.

## Contract and conventions

The backbone is not fine-tuned while the memory is built. Coreset sampling
starts from the ratio in `training_defaults.yaml`.

## Plan and acceptance criteria

- [ ] Adapter does not import FastAPI.
- [ ] Memory artifact path is outside Git.
- [ ] A tiny synthetic run distinguishes a constant image from a corrupted one.
- [ ] Shared interface documents score and map shapes.

## Tests and commands

```bash
.venv/bin/pytest backend/tests -q
```

## Non-goals

No PaDiM, no EfficientAD, no full MVTec benchmark (Wave 5).

## Evidence / handoff
