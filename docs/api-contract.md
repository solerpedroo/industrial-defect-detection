# API contract — Wave 7 comparison consumers

Base URL in local development: `http://localhost:8000`. The frontend reads `NEXT_PUBLIC_API_URL`.

## `GET /api/health`

```json
{
  "status": "ok",
  "api_version": "0.1.0"
}
```

## `GET /api/datasets`

Returns the MVTec catalog and per-category file counts from the local tree. `storage_root` is the repo-relative path `data/mvtec-ad`, never an absolute host path.

```json
{
  "id": "mvtec_ad",
  "name": "MVTec AD",
  "license": "CC BY-NC-SA 4.0",
  "license_url": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
  "commercial_use": false,
  "storage_root": "data/mvtec-ad",
  "categories": [
    {
      "id": "bottle",
      "name": "Bottle",
      "kind": "object",
      "purpose": "Detect changes on objects with defined geometry",
      "availability": "missing",
      "train_normal": 0,
      "validation_normal": 0,
      "test_normal": 0,
      "test_anomalous": 0,
      "mask_count": 0,
      "defect_types": [],
      "validation_split": "not_created"
    }
  ]
}
```

`availability` is `present` for a regular category directory; it is not an
integrity certificate. `validation_split` is `not_created`, `created`, or
`invalid`. A valid manifest at `ml/data/splits/<id>.json` makes `train_normal`
count retained fit images and `validation_normal` count held-out normal images.
Their sum is the official normal training inventory. Official test images are
never counted as validation.

Absent manifests expose all official training images as `train_normal` and zero
validation images. Malformed or stale manifests return `invalid`, the current
training inventory count and zero validation images; callers must not fit until
the offline audit and manifest creation succeed. Reads validate normal training
membership and file checksums, never decode test pixels, never repair manifests,
and never expose absolute host paths or checksum inventories.

## `GET /api/datasets/{category_id}/summary`

Same category object as one entry above. Unknown ids return `404`.

## `GET /api/models`

```json
{
  "models": [
    {
      "id": "autoencoder",
      "name": "Convolutional Autoencoder",
      "family": "reconstruction",
      "status": "planned",
      "backbone": null,
      "description": "Reconstruction model trained on normal images only.",
      "ready_categories": [],
      "synthetic_categories": []
    }
  ]
}
```

`status` is `planned` or `ready`. Without an installed serving manifest all models
remain `planned`. Startup-validated artifacts make a model `ready` for its
`ready_categories` only; another category can still return `501`.
`synthetic_categories` flags demonstration-only installations that require an
explicit server opt-in. Catalog IDs/descriptions remain YAML-owned. Neither
readiness nor a response claims real-data/CUDA acceptance.

## `GET /api/experiments`

Returns protocols, not measured metrics.

## `GET /api/experiments/{experiment_id}`

Returns one protocol. Unknown ids return `404`.

## `POST /api/inference`

Multipart fields:

- `image` — PNG or JPEG, verified by magic bytes, max 10 MiB and 4096 × 4096;
- `category_id` — a catalog category;
- `model_id` — a catalog model.

Unknown category or model: `422`. Bad image: `400`, `413`, or `415`. An uninstalled
category/model pair returns `501` with a detail string. Internal prediction/render
failure returns a sanitized `503`. The pre-parse multipart envelope is 10 MiB
plus 64 KiB overhead, including chunked uploads; exceeding it returns `413`.
Application payloads/results are not persisted or logged. Framework temporary
multipart spools close on every handled request path.

An installed, calibrated pair returns:

```json
{
  "model_id": "autoencoder",
  "category_id": "bottle",
  "score": 0.05,
  "label": "normal",
  "threshold": 0.1,
  "pixel_threshold": 0.1,
  "input_png": "<base64 PNG>",
  "anomaly_map_png": "<base64 PNG>",
  "overlay_png": "<base64 PNG>",
  "mask_png": "<base64 PNG>",
  "reconstruction_png": "<base64 PNG or null>",
  "display_max": 0.2,
  "map_width": 256,
  "map_height": 256,
  "data_kind": "mvtec_ad",
  "artifact_sha256": "<SHA-256>",
  "calibration_sha256": "<SHA-256>",
  "config_hash": "<model configuration SHA-256>",
  "inference_ms": 0.0,
  "request_ms": 0.0
}
```

`label` is `anomalous` exactly when raw `score > threshold`, otherwise `normal`.
The image threshold and pixel threshold come only from frozen normal-validation
calibration tied to the installed artifact/preprocessing/source/config. The mask
is a single-channel PNG of 0/255 from raw map `> pixel_threshold`; equality is
normal. Scores/maps have model-family-specific scales, not severity/probability.

All PNGs share model-resolution geometry, not original-image coordinates.
`input_png` is prepared display RGB. Autoencoder optionally supplies reconstruction;
PatchCore returns null. Heat colors have fixed [0,`display_max`] range, where max
is twice the frozen pixel threshold; zero threshold uses zero/nonzero intensity.
`overlay_png` blends prepared input and heat 50/50; UI opacity is client-side.
PNG download data URLs can be assembled from base64 without another API endpoint.

`inference_ms` measures synchronized cached prediction/preprocessing, excludes
image decode/PNG encode/queue waiting. `request_ms` includes decode, queued work
and rendering before response JSON serialization/network transfer. No checkpoint
load, calibration, dataset/test read or training occurs in a request. IDs/hashes
are exposed, never host paths or raw tensor values. `data_kind=synthetic` requires
server opt-in and must be visibly described as a demonstration.

## `POST /api/inference/compare`

Multipart fields: `image`, `category_id`, `model_a`, `model_b`. The two model IDs
must differ (`422` otherwise). Both category/model pairs must be installed (`501`
otherwise), checked before scoring either. The same decoded upload is used for
both; response is `{category_id, results: [InferenceResponse, InferenceResponse],
request_ms}`. Wave 7 displays these ephemeral results side by side, clears stale
requests, and disables identical/uninstalled choices. Saved-experiment comparison
instead reads a locally selected completed Wave 5 JSON in browser memory (16 MiB
limit); it adds no endpoint and never sends that report to the inference API.

## `GET /api/reports/{experiment_id}`

Returns a generated package when `REPORTS_ROOT/<experiment_id>/report.md` exists
(`REPORTS_ROOT` defaults to `sample_outputs/`):

```json
{
  "experiment_id": "exp-b-robustness",
  "markdown": "# Report: exp-b-robustness\n...",
  "figures": ["image_auroc.png"]
}
```

Unknown protocol ids return `404` and name the id. A known protocol without a
package returns `404` and says no artifact has been generated. Symlinked
packages are rejected. Markdown above 1 MiB returns `413`. The route does not
train, score, or embed PNG bytes. Figure files stay in the package directory.

Experiment B packages, when produced, reuse experiment A thresholds. Experiment
C packages may list a normal-train subset plan and an explicit blocked note
instead of AUROC. Reports state that anomaly scores are not physical severity.
