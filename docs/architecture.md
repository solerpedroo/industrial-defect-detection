# Architecture — Wave 3 offline autoencoder

## Boundary map

```text
Next.js UI  →  typed API client  →  FastAPI routes  →  catalogs / inference adapter  →  ml/
                 presentation         transport            contract                      numerical logic
```

The frontend never scores anomalies. FastAPI validates uploads, category ids, and model ids. `ml/` owns datasets, preprocessing and offline autoencoder training/prediction. PatchCore and metrics remain future work. HTTP handlers never start training.

## Components

1. **Web application** (`frontend/`) — Overview, Datasets, Inference Lab, Experiments, Comparison.
2. **Inference API** (`backend/`) — health, catalogs, and guarded prediction routes.
3. **Training pipeline** (`ml/`, `scripts/`, `experiments/`) — offline. Not invoked by HTTP.

There is no database. Experiment records will be YAML and JSON files.

## Image convention

Per ADR-004, decoded color images are contiguous RGB `numpy.ndarray` values with shape `(height, width, 3)`, `uint8`, range `[0, 255]`. Alpha is discarded at decode. Shared preprocessing creates CHW float32 arrays. Wave 3 converts them to PyTorch at the autoencoder boundary using model-specific unit-interval normalization.

## Catalogs

`ml/configs/categories.yaml` is the category contract: Bottle, Leather, Metal Nut, plus the MVTec license.  
`ml/configs/models.yaml` is the model contract: `autoencoder` and `patchcore`, both `planned`.  
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
adds actual offline autoencoder training and prediction consumers. PatchCore and
HTTP model consumers remain unimplemented. The user authorized engineering while
the licensed real-MVTec gate remains open; synthetic checks do not certify real
industrial detection performance.

## Offline autoencoder

PyTorch 2.10 is the optional `training` extra. Importing the web application does
not import torch. `ml/training/config.py` reads seed, batch size, epochs, patience,
learning rate and encoder widths from the canonical YAML. Autoencoder-specific
normalization is mean zero/std one in RGB unit-interval space, distinct from the
unchanged ImageNet defaults for the future feature backbone. Experimental overrides
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
The catalog and HTTP inference stay `planned`/501 until their serving gate.

## Inference boundary

`POST /api/inference` and `POST /api/inference/compare` accept multipart PNG or JPEG uploads. They reject unknown categories and models with `422`, invalid images with `400`, `413`, or `415`, and ready-looking calls with `501` until a wave marks the model `ready` and installs a loader.

Uploads are not written to disk.

## What each wave adds

| Wave | Addition |
| --- | --- |
| 1 | Implemented reader, integrity checks, normal validation manifests; real-data acceptance pending |
| 2 | Implemented shared preprocessing, mask geometry, provenance and previews; model consumers pending |
| 3 | Implemented offline PyTorch autoencoder, best checkpoints and reconstruction maps; CPU synthetic evidence |
| 4 | PatchCore adapter and shared `AnomalyModel` runtime |
| 5 | Metrics and validation thresholds |
| 6 | Heatmap overlay and downloads in Inference Lab |
| 7 | Comparison of saved experiment artifacts |
| 8 | Report export and experiments B and C |
| 9 | Deploy, demo, acceptance evidence |
