# MVP acceptance

The MVP is **not** complete. Wave 0 establishes the repository. Check these items only when the named wave has evidence.

Wave 1 code is implemented and independently reviewed with 72 Python tests,
3 frontend unit tests and 1 browser flow passing. Its normal-only split protocol
has synthetic evidence, including all three category IDs. A licensed local MVTec
copy is absent, so the two real-data acceptance items below remain open. See the
[`Wave 1 task card`](agentic/tasks/wave-1-dataset.md) and
[`real-data handoff`](agentic/handoffs/2026-10-14-wave-1-dataset.md).

Wave 2 engineering is implemented and reviewed: the full Python suite has
121 passing tests, including 49 preprocessing cases. The user authorized this
synthetic-data continuation while real-MVTec preparation remains open. A shared
stage-aware function and upload-decoder compatibility are verified. Wave 3 now
adds Autoencoder consumers, Wave 4 adds PatchCore, and Wave 6 adds optional HTTP
serving. See [`Wave 2 evidence`](agentic/tasks/wave-2-preprocessing.md).

Wave 3 integrates the shared pipeline into offline Autoencoder fitting and
prediction, with 170 Python tests passing (49 new model/training cases). It
implements normal-only Adam/MSE, early stopping, trusted best-checkpoint reload
and reconstruction maps. Evidence is synthetic CPU only; real-MVTec preparation,
CUDA and API serving remain open. Both model catalog entries remain `planned`.
See [`Wave 3 evidence`](agentic/tasks/wave-3-autoencoder.md).

Wave 4 implements frozen ResNet-18 PatchCore memory, seeded coreset selection,
trusted self-contained offline reload and a common score/map interface. The full
Python suite has 213 passing tests, including 43 PatchCore cases with the opt-in
actual pretrained smoke enabled. The smoke uses synthetic images, not MVTec.
Independent review found no blockers; upstream warnings remain visible. Raw
distances are not calibrated or comparable to Autoencoder MSE. Real-data, CUDA,
thresholds and HTTP readiness remain open. See
[`Wave 4 evidence`](agentic/tasks/wave-4-patchcore.md).

Wave 5 adds normal-validation calibration, image/pixel metrics including AUPRO,
and a JSON/CSV experiment A runner. Synthetic integration covers both actual
offline model implementations and all three category IDs. Thresholds freeze
before any test audit/read; hand-computed cases and an independent rank-metric
oracle validate numerical behavior. The full suite has 281 passing tests with
pretrained smoke enabled; 68 new evaluation cases pass independent review.
No real-MVTec results are claimed; CUDA,
HTTP serving and experiment A acceptance on licensed data remain open. See
[`Wave 5 evidence`](agentic/tasks/wave-5-evaluation.md).

Wave 6 implements startup-validated artifact/calibration serving, runtime category
readiness and upload-to-result UI with fixed-scale heatmap, mask/reconstruction,
opacity and PNG downloads. Evidence includes 38 runtime cases and three browser
flows using actual tiny synthetic models/API, plus independent review. Uninstalled
pairs retain 501; synthetic serving requires explicit opt-in and a visible warning.
Canonical model defaults remain planned. Real-MVTec quality and CUDA acceptance
remain open; synthetic HTTP success is not industrial readiness. See
[`Wave 6 evidence`](agentic/tasks/wave-6-inference-ui.md).

Wave 7 implements side-by-side results for one upload and a browser-only reader
of completed Wave 5 experiment JSON. Category-filtered metric tables preserve
undefined reasons and export recorded values/provenance as CSV. Reports are
structurally checked, not authenticated or numerically recomputed. Tests use
synthetic real-model API output and generated evaluator reports; real-MVTec
benchmark and CUDA acceptance remain open. See
[`Wave 7 evidence`](agentic/tasks/wave-7-comparison.md).

Wave 8 writes report packages and serves them from `GET /api/reports/{id}` when
the markdown artifact exists. Experiment B scoring reuses frozen experiment A
thresholds on in-memory degradations. Experiment C records subset membership
and an explicit block on retraining, because fit manifests still require the
full normal inventory. Scores are not described as physical severity. Real-data
and CUDA acceptance remain open. See [`Wave 8 evidence`](agentic/tasks/wave-8-reports.md).

Wave 9 documents an English reproduction path, a demo storyboard, and a deploy
path whose default image does not copy the dataset or checkpoints. A repository
test checks that Git tracks neither images nor weight files. Real-data, CUDA,
and experiment C retraining stay open. See [`Wave 9 evidence`](agentic/tasks/wave-9-ship.md).

| Criterion | Wave | Status |
| --- | --- | --- |
| Monorepo, catalogs, API shell, dashboard shell, CI | 0 | done |
| Bottle, Leather, and Metal Nut prepared locally | 1 | open |
| Train, validation, and official test isolated | 1 | open |
| Recorded preprocessing shared by train and inference | 2 | Both offline/HTTP model families integrated; real-data gate open |
| Autoencoder trains and writes reconstruction maps | 3 | implemented; real-data acceptance open |
| PatchCore builds a memory and writes anomaly maps | 4 | implemented; real-data acceptance open |
| Image-level and pixel-level metrics | 5 | implemented; real-MVTec experiment A open |
| Thresholds chosen without the official test split | 5 | implemented offline; real-data calibration open |
| UI loads an image and a compatible model | 6 | implemented; synthetic real-model browser evidence; real-data gate open |
| Dashboard shows quantitative and visual comparison | 7 | implemented; same-upload maps plus local saved metrics/CSV; synthetic evidence only |
| Results export as Markdown, CSV, JSON, or PNG | 8 | implemented offline; experiment C metrics blocked; real-data gate open |
| Essential modules covered by tests | 9 | local commands recorded on the Wave 9 card; real-data cases stay outside this row |
| README explains how to reproduce experiments | 9 | README quick start, offline evaluation, and report packages |
| Three comparative experiments documented | 8 | A measured offline; B scoring implemented; C membership plan only |

Reference: [`project-scope.md`](project-scope.md) §10.
