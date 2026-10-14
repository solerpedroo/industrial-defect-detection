# Scientific protocols

These directories record how experiments A, B, and C will be run. They do not
contain measured results yet.

| Protocol | Question | Config |
| --- | --- | --- |
| A | Autoencoder versus PatchCore on the same test ids | `autoencoder-vs-patchcore/` |
| B | Robustness under noise, blur, and brightness changes | `robustness/` |
| C | Performance against the amount of normal training data | `dataset-size/` |

Thresholds are chosen on validation data. The official MVTec test split is
reported after that choice is frozen. See `docs/project-scope.md` §8.
