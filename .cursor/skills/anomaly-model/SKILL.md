---
name: anomaly-model
description: Adds or changes an industrial anomaly model in this repository. Use for the autoencoder, PatchCore, PaDiM, or their shared prediction interface.
---

# Anomaly model workflow

Read `AGENTS.md`, the assigned task card, and `docs/agentic/architecture-decisions.md`.
If the split protocol or the tensor convention is missing, stop and propose an ADR.

## Required model contract

Document in the module docstring and tests:

- training images are normal only;
- input spatial size, channel order, dtype, and normalization;
- anomaly map shape and value range;
- what the image-level score aggregates;
- checkpoint path and the fact that it is not committed;
- whether the backbone weights are frozen.

Keep the implementation in `ml/models/<name>/`. The FastAPI adapter loads artifacts
and encodes PNG maps. The frontend displays the API payload.

## Implementation checklist

- [ ] Catalog entry and loader status change together.
- [ ] Training is a script or `ml/training/` function, never a route.
- [ ] Validation ids are excluded from the fit set.
- [ ] Synthetic test shows a changed image outscores a normal one, or states why that check waits for a later wave.
- [ ] Update the task card with commands and results.

## Numerical defaults

Starting hyperparameters live in `ml/configs/training_defaults.yaml`. They are
a hardware starting point, not a claim of optimality. Record the seed actually used.
