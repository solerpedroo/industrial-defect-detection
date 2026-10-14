---
name: evaluation-protocol
description: Adds metrics, thresholds, robustness runs, or dataset-size runs for industrial anomaly detection. Use for experiment A, B, or C and for image or pixel metrics.
---

# Evaluation protocol

Read ADR-003 and ADR-006 before computing a number that could enter a report.

## Rules

- Freeze the threshold on validation scores before opening the official test split.
- If validation contains only normal images, document the target false-positive rate and the quantile used.
- Use the same test ids for every model in a comparison.
- Write the config hash, seed, category, and model id next to the metric file.
- Degradation runs (experiment B) keep the same base filenames across conditions.
- Subset runs (experiment C) draw from normal training ids only.
- Do not describe AUROC, score, or mask area as physical severity.

## Metric set

Image: AUROC, average precision, precision, recall, F1, confusion counts.
Pixel: AUROC, average precision, AUPRO when implemented, IoU, Dice.
Compute: inference latency after warmup, prepare time, artifact size, approximate peak memory.

## Checklist

- [ ] Hand-computed synthetic case for any new metric.
- [ ] Threshold source recorded.
- [ ] Test ids identical across compared models.
- [ ] Results path gitignored when it embeds images.
