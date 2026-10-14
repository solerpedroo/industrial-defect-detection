# Deploy

Wave 9 owns production deploy. The split matches the local architecture.

## Frontend — Vercel

- Root directory: `frontend`
- Set `NEXT_PUBLIC_API_URL` to the public API origin, without a trailing slash.

## API — Railway

- Dockerfile: `backend/Dockerfile`
- The image installs the API only. It does not download MVTec AD and does not bundle checkpoints.
- Mount or copy prepared artifacts in a later wave. Do not commit them.
- Set `ALLOWED_ORIGINS` to the Vercel origin.

Training stays on a machine with the dataset and, when available, the GPU. The API serves artifacts produced by that offline run.
