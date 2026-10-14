# Experiment B — Robustness under image degradation

Question: how do controlled changes in image quality affect scores and maps?

Apply each condition to the same base test images and record the transform.

| Variation | Intensities |
| --- | --- |
| Gaussian noise | σ = 10, 20, 30 on 8-bit pixels |
| Gaussian blur | σ = 1, 2, 3 |
| Brightness | factors 0.6, 0.8, 1.2 |

Runs are produced by `ml/experiments/robustness.py`. Thresholds stay the ones
frozen on normal validation in experiment A. The original image is kept as a
reference condition. Degraded copies are not written back to the dataset.
Render a package with `scripts/render_report.py`.
