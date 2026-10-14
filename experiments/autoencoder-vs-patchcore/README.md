# Experiment A — Autoencoder vs PatchCore

Question: which algorithm identifies and localizes defects more accurately when
both see the same Bottle, Leather, and Metal Nut test images?

Compare image-level detection, pixel-level localization, artifact size, and
inference latency. Configuration differences must be written next to the metrics.
Do not retune on the official test split.

## Implemented offline protocol

Wave 5 implements the runner, not a completed real-MVTec benchmark. Synthetic
CPU fixtures exercise both real model consumers and all three category IDs;
their scores are not industrial accuracy evidence. Licensed data is still absent.

Use the [README evaluation commands](../../README.md#offline-evaluation--experiment-a).
The canonical YAML fixes targets before the official test report: normal image
FPR 0.05, pooled normal pixel FPR 0.01, higher quantile and strict `>` comparator.
Observed validation exceedance is recorded; population FPR is not guaranteed.
All category/model thresholds publish before the first test audit/read. The
normal split and preprocessing geometry must match both trusted artifacts.

Metrics: image AUROC, AP, precision/recall/F1, confusion counts/FPR/FNR; global
pixel AUROC/AP and micro IoU/Dice; equal-region 8-connected AUPRO integrated to
FPR 0.3 with a linearly interpolated endpoint, divided by 0.3. AP is the
non-interpolated precision-weighted recall increment, not trapezoidal PR area.
Metrics operate on nearest-neighbor resized ground truth at model resolution.
Undefined metrics are null with reasons. No per-model test-selected scaling or
threshold optimization is allowed. Both models consume exactly the same test IDs.

New ignored/external runs contain `calibration.json`, `results.csv` and completed
`results.json` published last. Completion requires the JSON flag; calibration
or CSV alone can belong to an interrupted run. Reports record config/seed, full
normal split, source/weight checksums, raw image scores, thresholds and environment,
without source images/maps. Use `--data-kind mvtec_ad` only for licensed MVTec;
that mode requires pretrained PatchCore. A declared synthetic run never closes
the benchmark gate. Partial category coverage is explicit; configuration overrides
are hashed even when all categories are covered.

Latency warms up on normal validation, synchronizes CUDA if selected and measures
prediction/preprocessing/output copy, excluding disk decode/model loading. Peak
CPU memory is process-lifetime high-water RSS, not isolated model memory; GPU
allocator peak is not whole-device memory. Artifact bytes and preparation time
are recorded; missing Autoencoder history makes duration explicitly unknown.
Exact metric pooling has an explicit count budget, not hidden subsampling or a
guarantee of available RAM. Freeze all protocol/model choices before running
the final official test; the CLI does not prevent deliberate reruns elsewhere.

## Real-data acceptance still required

Audit a licensed local Bottle, Leather and Metal Nut copy, build both compatible
artifacts with the fixed protocol, then report the real test once. No measured
real-data result table is committed yet. CUDA/VRAM acceptance also remains open.
