# Architecture — Wave 1 dataset pipeline

## Boundary map

```text
Next.js UI  →  typed API client  →  FastAPI routes  →  catalogs / inference adapter  →  ml/
                 presentation         transport            contract                      numerical logic
```

The frontend never scores anomalies. FastAPI validates uploads, category ids, and model ids. `ml/` will own datasets, preprocessing, models, training, and metrics. Wave 0 ships the boundaries and the catalogs. Training functions raise `NotImplementedError` until their wave.

## Components

1. **Web application** (`frontend/`) — Overview, Datasets, Inference Lab, Experiments, Comparison.
2. **Inference API** (`backend/`) — health, catalogs, and guarded prediction routes.
3. **Training pipeline** (`ml/`, `scripts/`, `experiments/`) — offline. Not invoked by HTTP.

There is no database. Experiment records will be YAML and JSON files.

## Image convention

Per ADR-004, decoded color images are contiguous RGB `numpy.ndarray` values with shape `(height, width, 3)`, `uint8`, range `[0, 255]`. Alpha is discarded at decode. Model tensors (resize, normalization) are created inside `ml/preprocessing/` in a later wave.

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

## Inference boundary

`POST /api/inference` and `POST /api/inference/compare` accept multipart PNG or JPEG uploads. They reject unknown categories and models with `422`, invalid images with `400`, `413`, or `415`, and ready-looking calls with `501` until a wave marks the model `ready` and installs a loader.

Uploads are not written to disk.

## What each wave adds

| Wave | Addition |
| --- | --- |
| 1 | Implemented reader, integrity checks, normal validation manifests; real-data acceptance pending |
| 2 | Recorded preprocessing and category-safe augmentation |
| 3 | PyTorch autoencoder, checkpoints, reconstruction maps |
| 4 | PatchCore adapter and shared `AnomalyModel` runtime |
| 5 | Metrics and validation thresholds |
| 6 | Heatmap overlay and downloads in Inference Lab |
| 7 | Comparison of saved experiment artifacts |
| 8 | Report export and experiments B and C |
| 9 | Deploy, demo, acceptance evidence |
