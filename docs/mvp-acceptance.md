# MVP acceptance

The MVP is **not** complete. Wave 0 establishes the repository. Check these items only when the named wave has evidence.

Wave 1 code is implemented and independently reviewed with 72 Python tests,
3 frontend unit tests and 1 browser flow passing. Its normal-only split protocol
has synthetic evidence, including all three category IDs. A licensed local MVTec
copy is absent, so the two real-data acceptance items below remain open. See the
[`Wave 1 task card`](agentic/tasks/wave-1-dataset.md) and
[`real-data handoff`](agentic/handoffs/2026-10-14-wave-1-dataset.md).

| Criterion | Wave | Status |
| --- | --- | --- |
| Monorepo, catalogs, API shell, dashboard shell, CI | 0 | done |
| Bottle, Leather, and Metal Nut prepared locally | 1 | open |
| Train, validation, and official test isolated | 1 | open |
| Recorded preprocessing shared by train and inference | 2 | open |
| Autoencoder trains and writes reconstruction maps | 3 | open |
| PatchCore builds a memory and writes anomaly maps | 4 | open |
| Image-level and pixel-level metrics | 5 | open |
| Thresholds chosen without the official test split | 5 | open |
| UI loads an image and a compatible model | 6 | open |
| Dashboard shows quantitative and visual comparison | 7 | open |
| Results export as Markdown, CSV, JSON, or PNG | 8 | open |
| Essential modules covered by tests | 9 | open |
| README explains how to reproduce experiments | 9 | open |
| Three comparative experiments documented | 8 | open |

Reference: [`project-scope.md`](project-scope.md) §10.
