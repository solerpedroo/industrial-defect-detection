# Architecture — Wave 7 comparison

## Boundary map

```text
Next.js UI  →  typed API client  →  FastAPI routes  →  catalogs / inference adapter  →  ml/
                 presentation         transport            contract                      numerical logic
```

The frontend never scores anomalies. FastAPI validates uploads, category ids, and model ids. `ml/` owns datasets, preprocessing, offline Autoencoder/PatchCore fitting/prediction and evaluation. HTTP handlers never start training.

## Components

1. **Web application** (`frontend/`) — Overview, Datasets, Inference Lab, Experiments, Comparison.
2. **Inference API** (`backend/`) — health, catalogs, and guarded prediction routes.
3. **Training pipeline** (`ml/`, `scripts/`, `experiments/`) — offline. Not invoked by HTTP.

There is no database. Experiment records will be YAML and JSON files.

## Image convention

Per ADR-004, decoded color images are contiguous RGB `numpy.ndarray` values with shape `(height, width, 3)`, `uint8`, range `[0, 255]`. Alpha is discarded at decode. Shared preprocessing creates CHW float32 arrays. Waves 3/4 convert them to PyTorch at model boundaries using unit-interval normalization for Autoencoder and ImageNet normalization for PatchCore.

## Catalogs

`ml/configs/categories.yaml` is the category contract: Bottle, Leather, Metal Nut, plus the MVTec license.  
`ml/configs/models.yaml` owns `autoencoder` and `patchcore`, defaulting to `planned`.
HTTP readiness is derived at startup for explicitly installed category/model pairs.
`ml/configs/training_defaults.yaml` records the starting hyperparameters. They are not claimed to be optimal.

`GET /api/datasets` and `GET /api/models` serve those files. When `data/mvtec-ad/<category>` exists locally, the summary counts train, test, and mask files. Counts stay zero and `availability` is `missing` until the dataset is downloaded outside Git.

## Dataset loader and normal validation split

`ml/datasets/reader.py` audits the standard MVTec category tree and returns
root-relative image IDs, defect types, optional test mask IDs, and structured
integrity issues. It verifies and fully decodes PNG/JPEG/BMP/TIFF images without
resizing. Masks must match the anomalous test image stem and resolution and
contain single-channel 0/255 values. Unexpected paths, symlinks, special files,
unreadable images and missing/orphan/ambiguous masks fail the audit. Source files
are never moved or deleted. RGB pixel hashes including spatial dimensions detect
duplicates within train or across train/test, including different lossless
encodings. Repetition entirely within test does not select the normal holdout.

`ml/datasets/splits.py` audits first and writes a versioned JSON manifest to
`ml/data/splits/<category>.json`. Seed plus image ID are ranked by SHA-256; the
first `ceil(n * validation_fraction)` IDs form validation. Both fit and validation
must be nonempty. All candidates come from `train/good`. Byte-identical manifests
can be recreated with the same inputs. They record the full normal training
checksums, seed, fraction, retained fit IDs and held-out IDs, without test labels,
image payloads or absolute dataset paths. Writes are atomic per category;
replacement requires `--overwrite`. A multi-category CLI invocation audits all
categories and checks existing manifests and partition sizes before writing; a
later filesystem error may still leave earlier categories successfully written,
which is reported with a nonzero exit and per-category statuses.

`load_split_manifest` checks schema, deterministic membership, the training tree
and current raw checksums. `partition_samples(root, category, "train")` and its
`"validation"` variant require a valid manifest and never open official test
images. The `"test"` variant returns audited image/mask pairs for later evaluation.
Returned samples are metadata; tensor preprocessing belongs to Wave 2. Treat
the dataset as immutable during audit and consumption. Checksum verification
detects drift on load but is not a filesystem snapshot during future training.

`ml/datasets/summary.py` reports `validation_normal` and the manifest status in
the existing API summary. `created` subtracts held-out IDs from `train_normal`;
`invalid` exposes zero validation images and warns the UI. Summary counts are
inventory counts, not a substitute for the offline integrity audit. Summary
reads hash normal training files for freshness and list test metadata only.
This correctness-first approach has file I/O proportional to normal training
size; it does not cache an obsolete validity decision.

Run `.venv/bin/python scripts/validate_dataset.py` for an audit (exit 1 when any
catalog category is absent or invalid). Add `--create-split
--validation-fraction 0.2` to generate manifests. `--category` is repeatable,
`--dataset-root` and `--manifest-dir` support offline copies, and `--json` emits
the structured report. Default categories/root and seed come from canonical YAML.
The API uses the repository's default dataset and manifest locations.

## Shared preprocessing

`ml/preprocessing/config.py` loads the configured size, RGB mean/std and
preprocessing defaults from `training_defaults.yaml`. Per-category
`preprocessing.allow_horizontal_flip` permissions live in `categories.yaml`.
All initial categories deny flipping. The loader rejects unknown categories,
malformed statistics and attempts to enable unapproved flips. Explicit experiment
configurations can be constructed or derived with `dataclasses.replace`; those
callers are responsible for preserving the reviewed category permission. Effective
settings, including that permission, are recorded and hashed.

`apply_preprocess(image, config, stage=...)` is the only numerical path for
`train`, `validation`, `test` and `inference`. It accepts RGB HWC uint8 [0, 255]
and returns `PreprocessResult`:

- `model_input`: contiguous `(3, size, size)` float32 values, computed as
  `(RGB / 255 - mean) / std`. These values can be negative or exceed one.
- `display_rgb`: square HWC uint8 RGB after resize/augmentation/grayscale, before
  normalization, for display or reconstruction comparisons.
- `mask`: optional square H×W uint8 0/255 labels, synchronized with image geometry.
- `record`: JSON-safe effective configuration, schema version, SHA-256 config hash,
  source/output shape and dtype, stage/seed, ordered operations and library versions.

Images use bilinear resize; masks use nearest-neighbor. Optional Pillow L
grayscale is replicated to RGB before normalization; different channel statistics
can still produce different normalized channels. Resizing stretches the original
aspect ratio to the configured square; it does not crop or pad. The record makes
this spatial choice explicit. Inputs are never modified, including strided arrays.

Only horizontal flipping is currently supported. Train augmentation requires an
explicit seed when probability is positive and category permission is enabled.
The local RNG draws one image/mask decision. Validation, test and inference never
augment and reject training seeds. A later training loop must derive reproducible
per-sample/per-epoch seeds; reusing one seed on every call repeats its decision.
The top-level YAML seed is an experiment seed, not a hidden global RNG mutation.

`preprocess_preview` calls the same function once and returns independent
before/after display RGB arrays and masks with the same record. No PNG encoding,
HTTP route or preview UI is added in this wave. To use the pipeline:

```python
from ml.preprocessing import apply_preprocess, load_preprocess_config

config = load_preprocess_config("bottle")
result = apply_preprocess(rgb_uint8, config, stage="inference")
model_input = result.model_input  # Model boundaries call torch.from_numpy.
metadata = result.record         # Persist beside experiment/checkpoint artifacts.
```

The shared paths have synthetic and upload-decoder integration evidence. Wave 3
adds actual offline autoencoder training and prediction consumers; Wave 4 adds
PatchCore. Wave 6 adds optional calibrated HTTP consumers. The user authorized engineering while
the licensed real-MVTec gate remains open; synthetic checks do not certify real
industrial detection performance.

## Offline autoencoder

PyTorch 2.10 is the optional `training` extra. Importing the web application does
not import torch. `ml/training/config.py` reads seed, batch size, epochs, patience,
learning rate and encoder widths from the canonical YAML. Autoencoder-specific
normalization is mean zero/std one in RGB unit-interval space, distinct from the
unchanged ImageNet defaults for the PatchCore feature backbone. Experimental overrides
are recorded and cannot override the canonical category flip permission in fit.

`ConvolutionalAutoencoder` has three stride-2 Conv2d/ReLU blocks and three mirrored
ConvTranspose2d blocks with a final sigmoid. Initialization is explicitly CPU
float32, independently of caller torch defaults. All weights train; there is no
pretrained/frozen backbone. BCHW RGB float32 [0,1] inputs need dimensions divisible
by eight. The training square is at most 4096 and widths are integers in [1,256].

`ml.training.autoencoder.train_autoencoder` loads the Wave 1 manifest, then reads
only its retained normal fit IDs and held-out normal validation IDs. Each source
read hashes the same bytes it decodes. Normal images pass through `apply_preprocess`
and `torch.from_numpy`. Fit uses Adam/MSE; validation uses inference_mode and no
augmentation. Losses are sample-weighted, including partial final batches. Strict
validation improvement saves independent best-state snapshots; patience counts
consecutive non-improvements. Neither fitting nor selection opens official test
images or masks. The final manifest recheck rejects content/inventory drift.

Initialization/shuffle are seeded; enabled augmentation derives seeds from the
first four big-endian bytes of `SHA256(seed:epoch:image_id)`. Local loader generators
and restored torch RNG/backend settings prevent leaking randomness into callers.
Deterministic algorithms are requested, but bit identity across hardware/library
versions is not claimed. Regression evidence is CPU only; CUDA is a supported
device path with explicit availability validation, not a verified performance claim.

Every run requires a new directory beneath ignored `checkpoints/`, `artifacts/`
or `runs/`, or an explicit external directory outside the source dataset. Existing
runs and tracked repository roots are rejected. Artifacts are `best.pt` and
`history.json`. They record effective config/hashes, full normal manifest provenance,
seed derivation, sample-weighted losses, best epoch, versions, device/hardware and
duration. Prepared weights are staged and hashed; final history is published first,
then completed `best.pt` last. Its complete flag is the readiness marker. On failure,
earlier incomplete best weights may remain for diagnosis and must not be served.
These artifacts support inference reload, not optimizer/scheduler training resume.

`load_autoencoder(checkpoint, trusted_root=..., device=...)` rejects escaping,
symlinked and special paths, temporary staging files, incomplete runs, incompatible
schemas/config hashes or invalid weight shapes/dtypes/values. It deserializes with
`weights_only=True` on CPU, initializes without consuming caller RNG, then transfers
to CPU/CUDA. The caller must explicitly select a trusted artifact directory; restricted
loading is not permission to accept uploaded checkpoints.

`AutoencoderPredictor.predict(rgb_uint8)` uses the same recorded unit-interval
pipeline, eval and inference_mode. It returns square HWC float32 reconstruction
[0,1], HW float32 channel-mean squared RGB difference [0,1], the map's spatial mean
as score, prepared display RGB and preprocessing provenance. These are at model
resolution, not original-resolution localization. There is no threshold or label.
The canonical catalog defaults to `planned`; Wave 6 installs compatible runtime
category/model pairs without claiming industrial acceptance.

## Offline PatchCore

ADR-011 adds optional Anomalib 2.7.0, torchvision 0.25.x and timm 1.0.x with
torch 2.10.x. The low-level `PatchcoreModel` avoids Lightning's dataset,
postprocessing and calibration paths. Importing the API still imports neither
torch nor Anomalib. `PatchcoreConfig` loads canonical defaults and records all
overrides. ResNet-18 layer2/layer3 is pretrained by default; random initialization
is an explicitly recorded regression-only option. Shared preprocessing resizes
RGB to a square multiple of 32 and applies ImageNet mean/std. Fit augmentation
is disabled so memory construction and prediction use the same transform.

`build_patchcore_memory` checks a Wave 1 manifest and decodes only retained normal
fit IDs. It hashes the exact bytes decoded, extracts pooled concatenated frozen
features and builds Anomalib's seeded k-center coreset. Backbone parameters never
require gradients; its feature extractor stays eval, including BatchNorm, while
the model's training-mode forward collects embeddings. A before/after backbone
digest checks that memory building did not change pretrained state. Validation
normal files are hashed during manifest checks, but their pixels are not decoded;
official test images/masks are never opened. Final source checks reject drift.

Preflight patch count must fit the explicit budget, and
`floor(patch_count * coreset_sampling_ratio) >= num_neighbors`. Neither the ratio
nor fit selection silently changes to fit available resources. The count budget
does not guarantee memory/time capacity: feature storage, projected distances
and iterative k-center selection can be expensive. Seeded torch, NumPy and Python
RNGs, float32 default dtype and deterministic/backend flags are scoped/restored.
Repeatability is verified within one CPU/library environment, not across devices
or releases. CUDA availability is checked, but CUDA execution remains unverified.

Every run uses a new ignored/external directory outside the dataset. `memory.pt`
contains the entire extracted backbone state, coreset and primitive provenance.
Metadata includes config/preprocessing hashes, the normal split/checksums,
patch/memory counts and dimension, pretrained source/backbone digest, seed,
versions, device/hardware and duration. The completed checkpoint is staged;
`metadata.json` with its checksum publishes first, then `memory.pt` last. Failure
does not publish a completed checkpoint. The directory can remain for diagnosis.

`load_patchcore(..., trusted_root=...)` restricts regular non-symlink `.pt` paths
under an explicit trusted root, uses CPU `weights_only=True`, and checks schema,
completion/config hash, memory dimensions/count, state keys/shapes/dtypes,
finiteness and backbone digest. It reconstructs the backend with pretrained
retrieval disabled and restores stored weights/memory. Artifacts are locally
trusted inputs, not a format for uploaded untrusted checkpoints.

Both predictors implement `AnomalyPredictor.predict(RGB HWC uint8)` and return
`AnomalyPrediction`: a finite nonnegative scalar and square HW float32 map at
model resolution. PatchCore uses raw feature distances without an upper bound.
Its score is Anomalib's weighted maximum-patch nearest-neighbor score (plain
maximum for one neighbor); its map is interpolated/smoothed, so score is not the
map mean or necessarily its maximum. Autoencoder uses mean squared error. These
raw scales cannot be compared directly or interpreted as severity/probability.
No threshold, label, calibration or HTTP readiness is added in Wave 4.

Synthetic CPU evidence covers the real library, leakage guards, frozen state,
reproducibility, publication failures and offline reload. An optional smoke also
loads actual pretrained weights; CI's default tests use explicit random weights
without network retrieval. Upstream TorchScript/Python 3.14 and deprecated-import
warnings are retained. Real-MVTec benchmark and CUDA acceptance remain open.

## Validation-frozen experiment A

ADR-012 defines one dependency-free NumPy metric/calibration path. Normal held-out
image scores and pooled normal map pixels get separate higher empirical quantiles
at `1 - target_fpr`. Decisions use strictly `>`; observed calibration exceedance,
quantile/method, counts, source IDs/checksums and model config/artifact hashes are
recorded. This is not a guarantee of unseen false-positive rates or independent
pixel trials. Neither model's raw values are rescaled to another model's range.

`ml.evaluation.evaluate` lazily invokes the offline experiment runner. It loads
both trusted artifacts against the current normal manifest, category, seed,
resolution and shared inference geometry. All requested category/model thresholds
are frozen and written to `calibration.json` before any test inventory/audit/read.
Only then does it audit the source, prepare identical test IDs/nearest-neighbor
resized masks and evaluate each model. Test images/masks and model files have
byte hashes and drift checks. Full normal split provenance remains in the report.

Image AUROC uses grouped tied-score ROC trapezoids. `aupr` fields mean
non-interpolated average precision, not trapezoidal PR area. Pixel rank metrics
pool every evaluated model-resolution pixel; IoU/Dice use global micro counts.
AUPRO gives equal weight to each 8-connected foreground region, pools background
FPR, clips the curve to the configured FPR limit with linear interpolation, then
normalizes area by that limit. Undefined denominators/classes/regions yield JSON
null and reasons. Lossy conversion of accepted scores to float64 is rejected;
large integer values above 2**53 cannot silently turn into ties.

Protocol defaults (target FPRs, AUPRO limit, warmup, count budget) live in
`experiments/autoencoder-vs-patchcore/config.yaml`. Config changes are hashed.
Exact pooled metrics are bounded by an explicit per-category validation/test
pixel count, without sampling; sorting and region labeling may still consume
substantial RAM/time. Count budget is not a resource-capacity guarantee.

Timing warms up on a normal validation image, synchronizes CUDA around each
prediction and excludes model load/file decode. It includes preprocessing and
output validation/copy. Reports include mean/median/p95, per-image latency,
throughput, artifact bytes, preparation duration and environment. CPU peak is
approximate process-lifetime high-water RSS, not per-model consumption. CUDA
peak is allocator memory including the loaded model, not whole-device usage.
Autoencoder duration comes from a complete `history.json` bound to checkpoint
SHA/config/split; absence yields an explicit unknown, invalid/drifted history
fails. PatchCore duration lives inside its hashed checkpoint metadata.

Runs require a new ignored/external directory outside the dataset. Frozen
calibration is incomplete by design; CSV publishes before completed JSON, the
final completion marker. Reports contain numbers/IDs/hashes, not image payloads.
`data_kind` explicitly declares synthetic or MVTec; real experiment A rejects
random PatchCore weights. Category coverage is marked partial/complete independent
of requested ordering, while all overrides remain recorded. Synthetic three-
category CPU tests are engineering evidence, not real-MVTec benchmark results.
No calibrated HTTP loader, UI heatmap or catalog readiness is added here.

## Inference boundary

ADR-013 implements an optional operator-owned `INFERENCE_MANIFEST` installation.
It lists an explicit trusted root, CPU/CUDA device and relative checkpoint/
Wave 5 frozen `calibration.json` paths for each category/model pair. Startup
rejects unsafe regular-file paths, unsupported schema, checkpoint/calibration
hash drift, incompatible model/config/preprocessing hashes, normal split/held-out
membership/checksums and quantile/comparator/count provenance. No raw dataset
files are needed. Predictors are loaded, frozen, warmed and cached before serving;
restart to change a snapshot. Family imports are lazy, so an Autoencoder-only
installation needs `training`, not Anomalib. Without installation neither torch
nor Anomalib imports through the API.

Runtime `GET /api/models` overlays ready status, `ready_categories` and
`synthetic_categories` on canonical YAML IDs. A globally ready model can still
have an uninstalled category. Synthetic calibration requires the server-only
`INFERENCE_ALLOW_SYNTHETIC=1` opt-in and a response/UI demonstration warning.
Real MVTec/CUDA readiness is not inferred from passing synthetic serving tests.

Both existing inference POST routes validate category/model IDs without touching
dataset sources. Unknown IDs return 422; invalid images return 400/413/415;
uninstalled pairs retain 501; invalid model output returns sanitized 503. The
ASGI transport bounds complete/chunked multipart bodies before parsing to 10 MiB
plus 64 KiB overhead. UploadFile reads additionally stop at limit+1 and always
close. Framework multipart spools may temporarily use disk; application uploads,
PNGs and predictions are not persisted/logged. Decode, prediction and encoding
run off the async event loop. Predictor use is serialized per process to avoid
concurrent mutable backend/device work; workers each own their loaded copies.

Prediction uses the common RGB/model-square contract. Image label is raw score
`> image_threshold`; mask is raw map `> pixel_threshold` as 0/255 PNG. Prepared
input, colored heatmap, mask, 50% overlay and optional Autoencoder reconstruction
are base64 PNGs at the same model resolution, not original coordinates. Heat
colors use fixed [0,2×pixel threshold], with zero/nonzero intensity for threshold
zero. Display saturation does not normalize raw scores or imply severity.
The existing compare transport preflights both installed pairs and reuses one
decoded image. Wave 7 consumes this transport for live side-by-side results.

`inference_ms` covers synchronized prediction/preprocessing, not decode/encode,
load or queue waiting. `request_ms` includes decode/queued work/PNG rendering
before serialization/network transfer. HTTP never trains or recalibrates, opens
test pixels/masks or reads weight files per request. Response hashes identify
the installed model/config/calibration, without exposing absolute host paths.

Inference Lab consumes runtime category coverage, derives selections from API
catalogs, submits explicitly and aborts/ignores obsolete responses. Input changes,
resubmissions and failures clear old PNGs/downloads. Opacity changes are client-
only. Synthetic warnings, independent image/pixel decisions and score/geometry
limitations are explicit. Playwright tests exercise actual models/API, PNG
downloads, cleanup and stale response guards; production-test API URLs must be
supplied at build time. Existing services are preserved through alternate ports.

## Wave 7 comparison boundary

`frontend/features/comparison/` displays two installed model results from the same
upload, with synchronized overlay opacity, calibrated decisions, latency and
per-model provenance. It derives selections/readiness from catalogs, rejects
identical choices and aborts/ignores obsolete responses. Heat colors/scores retain
model-specific scales and model-resolution geometry.

`frontend/features/experiments/report.ts` projects a locally selected completed
Wave 5 JSON (16 MiB maximum) into a read-only metric table. It validates completion,
schema, catalog IDs, coverage, shared ordered test IDs/labels, finite values/null
reasons and calibration consistency. No numerical computation, checkpoint or
dataset access occurs. Validation is structural, not provenance authentication.
Reports stay in browser memory; replacement/error/unmount invalidates older reads.
Saved categories and installed live categories are independent.

The table exposes test split, synthetic/partial scope, seed, resolution, warmup,
device and hashes; CSV exports only the selected category with recorded values,
null reasons, model/artifact identity and operating points. CSV fields are escaped
and formula prefixes neutralized. No dependency, endpoint or persistence was added.
Actual tiny synthetic evaluation reports and real-model API results exercise both
browser paths. Real-data scientific acceptance remains open (ADR-014).

## Wave 8 reports

ADR-015 adds an offline package writer in `ml/experiments/render.py` and
`scripts/render_report.py`. A package is `report.md`, `results.json`,
`results.csv` and, when recorded image AUROC values exist, `image_auroc.png`.
`GET /api/reports/{id}` returns the markdown and PNG filenames from
`REPORTS_ROOT` (default `sample_outputs/`). It does not train or score.

Experiment B (`ml/experiments/robustness.py`) copies each base image, applies
the recorded noise, blur and brightness grid plus an original reference, and
scores every condition with thresholds copied from a completed experiment A
report. Degraded pixels are not written to the dataset. Experiment C
(`ml/experiments/subsets.py`) records seeded 25/50/75/100 percent draws of
normal train ids. Those draws are not fit manifests: the split loader still
requires the full normal inventory, so the report states that retraining is
blocked and does not invent AUROC.

Packages always say that scores and mask area are not physical severity.
Missing measurements use an explicit blocked section.

## Wave 9 ship boundary

ADR-016 keeps MVTec images and checkpoints out of Git and out of
`backend/Dockerfile`. The README sequences local training, experiment A,
report packages, and optional HTTP serving. `docs/deploy.md` describes Vercel
and Railway for the catalog UI and API. `docs/demo-storyboard.md` limits what
a short recording may show. None of these artifacts close real-data or CUDA
acceptance.

## What each wave adds

| Wave | Addition |
| --- | --- |
| 1 | Implemented reader, integrity checks, normal validation manifests; real-data acceptance pending |
| 2 | Implemented shared preprocessing, mask geometry, provenance and previews; model consumers pending |
| 3 | Implemented offline PyTorch autoencoder, best checkpoints and reconstruction maps; CPU synthetic evidence |
| 4 | Implemented frozen PatchCore adapter, memory artifacts and shared `AnomalyPredictor`; CPU synthetic/pretrained smoke evidence |
| 5 | Implemented exact metrics, normal-validation thresholds and offline JSON/CSV experiment A; real-MVTec benchmark pending |
| 6 | Implemented trusted calibrated HTTP serving, heatmap/overlay/mask/reconstruction and downloads; synthetic real-model browser evidence |
| 7 | Implemented same-upload model comparison and local saved metrics/category filter/CSV; synthetic browser evidence |
| 8 | Report packages, frozen-threshold experiment B scoring, and an experiment C membership plan; subset retraining blocked |
| 9 | README reproduction path, dataset-free deploy notes, demo storyboard; real-data and CUDA gates stay open |
