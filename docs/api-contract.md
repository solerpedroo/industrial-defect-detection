# API contract — Wave 0

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
      "test_normal": 0,
      "test_anomalous": 0,
      "mask_count": 0,
      "defect_types": [],
      "validation_split": "not_created"
    }
  ]
}
```

`availability` is `present` when `data/mvtec-ad/<id>` exists. `validation_split` stays `not_created` until Wave 1 writes a split manifest. Official test images are never counted as validation.

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
      "description": "Reconstruction model trained on normal images only."
    }
  ]
}
```

`status` is `planned` or `ready`. Wave 0 has no `ready` model.

## `GET /api/experiments`

Returns protocols, not measured metrics.

## `GET /api/experiments/{experiment_id}`

Returns one protocol. Unknown ids return `404`.

## `POST /api/inference`

Multipart fields:

- `image` — PNG or JPEG, verified by magic bytes, max 10 MiB and 4096 × 4096;
- `category_id` — a catalog category;
- `model_id` — a catalog model.

Unknown category or model: `422`. Bad image: `400`, `413`, or `415`. Valid input for a `planned` model: `501` with a detail string. The body is not stored.

When a later wave implements inference, the JSON body will be:

```json
{
  "model_id": "autoencoder",
  "category_id": "bottle",
  "score": 0.0,
  "label": "anomalous",
  "threshold": 0.0,
  "anomaly_map_png": "<base64 PNG>",
  "overlay_png": "<base64 PNG>",
  "inference_ms": 0.0,
  "request_ms": 0.0
}
```

`label` depends on a threshold chosen on validation data. `score` is not a physical severity.

## `POST /api/inference/compare`

Multipart fields: `image`, `category_id`, `model_a`, `model_b`. The two model ids must differ. Wave 0 returns the same validation errors, or `501` when both ids exist.

## `GET /api/reports/{experiment_id}`

`404` until Wave 8 writes a report artifact. Unknown experiment ids also return `404`, with a detail that names the missing id.
