# Deploy

The published path is the Next.js UI plus the catalog API. It does not publish
MVTec AD and does not bake checkpoints into Git or the default image. This
document does not record a live Vercel or Railway deployment.

## What stays off the image

`backend/Dockerfile` copies `backend/pyproject.toml`, `backend/app`, and `ml`.
It installs the API extra only. It does not copy `data/`, `checkpoints/`,
`runs/`, `sample_outputs/`, or weight files. `docker-compose.yml` publishes
port 8000 and sets `ALLOWED_ORIGINS`. It does not mount the dataset.

Manifests, checkpoints, and report packages stay on a machine you control.
Mount them as volumes outside this repository. Do not commit them to finish
a deploy.

## Frontend — Vercel

- Root directory: `frontend`
- Set `NEXT_PUBLIC_API_URL` to the public API origin, without a trailing slash.
- Do not put `INFERENCE_MANIFEST`, dataset paths, or checkpoint paths in any
  `NEXT_PUBLIC_*` variable.

## API — Railway

- Dockerfile: `backend/Dockerfile`
- Set `ALLOWED_ORIGINS` to the Vercel origin.
- Leave `INFERENCE_MANIFEST` unset to serve catalogs with models `planned`.
- Model serving needs `backend[training]` or `backend[patchcore]` in an
  operator image that is not this default Dockerfile, plus a manifest outside
  the repository. Synthetic calibration also needs `INFERENCE_ALLOW_SYNTHETIC=1`
  and must be labeled as a demonstration.
- Optional `REPORTS_ROOT` points at packages from `scripts/render_report.py`.
  The API returns Markdown and PNG names. It does not return dataset bytes.

Training stays on a machine that has the licensed data and, when you have
measured it, a GPU. The default image does not download MVTec AD.
