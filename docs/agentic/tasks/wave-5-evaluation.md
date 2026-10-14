# Task: `wave-5-evaluation`

- **Status:** proposed
- **Wave:** 5 — Evaluation and benchmark
- **Owner / role:** unassigned
- **Model or category IDs:** `autoencoder`, `patchcore`
- **Depends on:** Waves 3 and 4
- **Allowed files:** `ml/evaluation/`, `experiments/autoencoder-vs-patchcore/`, metric tests.

## Goal

Compute image-level and pixel-level metrics on the official test split after
thresholds are frozen on validation. Run experiment A on Bottle, Leather, and
Metal Nut.

## Contract and conventions

ADR-003 and ADR-006. Record latency with warmup. Record approximate peak memory
and artifact size. Export CSV and JSON under the experiment directory, gitignored
if they embed images.

## Plan and acceptance criteria

- [ ] Image AUROC, AUPR, precision, recall, F1, confusion counts.
- [ ] Pixel AUROC, AUPR, IoU, Dice, and AUPRO when the implementation is available.
- [ ] Threshold source is written next to the metrics.
- [ ] Same test ids for both models.

## Tests and commands

```bash
.venv/bin/pytest backend/tests -q
```

Synthetic score arrays must match hand-computed AUROC cases.

## Non-goals

No degradation study (experiment B) and no subset study (experiment C).

## Evidence / handoff
