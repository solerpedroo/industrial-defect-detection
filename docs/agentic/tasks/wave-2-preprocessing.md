# Task: `wave-2-preprocessing`

- **Status:** proposed
- **Wave:** 2 — Image preprocessing
- **Owner / role:** unassigned
- **Model or category IDs:** all catalog categories
- **Depends on:** Wave 1
- **Allowed files:** `ml/preprocessing/`, `ml/configs/training_defaults.yaml`, preprocess tests.

## Goal

Provide one recorded preprocess path for training and inference: resize to the
configured square size, RGB normalization, optional grayscale, and a log of
the operations applied.

## Contract and conventions

ADR-004. Augmentations are opt-in per category. Horizontal flips are disabled
unless the category config allows them. Geometric changes that would move a
defect mask must be applied to the mask with the same parameters.

## Plan and acceptance criteria

- [ ] Config dataclass matches `training_defaults.yaml` image size.
- [ ] Before/after preview helper exists for the future UI.
- [ ] Train and inference call the same function.

## Tests and commands

```bash
.venv/bin/pytest backend/tests -q
```

## Non-goals

No training loop and no automatic claim that every augmentation is valid.

## Evidence / handoff
