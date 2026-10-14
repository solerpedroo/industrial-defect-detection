# Architecture — Wave 0 baseline

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

## Inference boundary

`POST /api/inference` and `POST /api/inference/compare` accept multipart PNG or JPEG uploads. They reject unknown categories and models with `422`, invalid images with `400`, `413`, or `415`, and ready-looking calls with `501` until a wave marks the model `ready` and installs a loader.

Uploads are not written to disk.

## What later waves add

| Wave | Addition |
| --- | --- |
| 1 | MVTec reader, integrity checks, validation split carved from normal training images |
| 2 | Recorded preprocessing and category-safe augmentation |
| 3 | PyTorch autoencoder, checkpoints, reconstruction maps |
| 4 | PatchCore adapter and shared `AnomalyModel` runtime |
| 5 | Metrics and validation thresholds |
| 6 | Heatmap overlay and downloads in Inference Lab |
| 7 | Comparison of saved experiment artifacts |
| 8 | Report export and experiments B and C |
| 9 | Deploy, demo, acceptance evidence |
