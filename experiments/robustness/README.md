# Experiment B — Robustness under image degradation

Question: how do controlled changes in image quality affect scores and maps?

Apply each condition to the same base test images and record the transform.

| Variation | Intensities |
| --- | --- |
| Gaussian noise | σ = 10, 20, 30 on 8-bit pixels |
| Gaussian blur | σ = 1, 2, 3 |
| Brightness | factors 0.6, 0.8, 1.2 |

Runs are produced in Wave 8.
