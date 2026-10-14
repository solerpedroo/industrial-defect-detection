# Industrial Defect Detection

Inspection system for **unsupervised visual anomaly detection** on industrial
images: a convolutional autoencoder and PatchCore, pixel-level localization,
and reproducible comparisons on MVTec AD.

**Stack:** Next.js (TypeScript) · FastAPI · PyTorch (from Wave 3) · Anomalib (from Wave 4)  
**Status:** Wave 1 dataset pipeline implemented and tested on synthetic trees.
Real MVTec preparation is pending. Training and live heatmaps start in later
waves. See [`docs/mvp-acceptance.md`](docs/mvp-acceptance.md).

## What this repository does today

- Catalogs Bottle, Leather, and Metal Nut plus the two model families
- Audits local MVTec images, mask pairs, corrupt files and duplicate content
- Creates reproducible normal-only validation manifests with source checksums
- Shows fit, validation, test and mask counts plus split validity in Datasets
- Validates inspection uploads and reports that no checkpoint is ready yet
- Dashboard pages: Overview, Datasets, Inference Lab, Experiments, Comparison

Training does not run inside the API. Measured metrics, heatmaps, and reports
arrive in later waves. Protocols A, B, and C are specified under [`experiments/`](experiments/README.md).

## Architecture

```text
Browser (Next.js)  →  /api/*  →  FastAPI  →  YAML catalogs
                                      ↓
                               ml/  (datasets, models, metrics — offline)
```

Details: [`docs/architecture.md`](docs/architecture.md) · API: [`docs/api-contract.md`](docs/api-contract.md)

## Quick start

**Requirements:** Python 3.11+, Node 24+

```bash
python -m venv .venv
.venv/bin/python -m pip install -e "./backend[dev]"

# Terminal 1 — API
.venv/bin/uvicorn app.main:app --app-dir backend --reload --port 8000

# Terminal 2 — UI
cd frontend
cp .env.example .env.local
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

Optional API container:

```bash
docker compose up --build
```

MVTec AD is CC BY-NC-SA 4.0 and is not included. Download it separately and
extract it to `data/mvtec-ad/`, then run:

```bash
.venv/bin/python scripts/validate_dataset.py
.venv/bin/python scripts/validate_dataset.py --create-split --validation-fraction 0.2
```

The audit returns a nonzero exit for missing/invalid data. The fraction is an
explicit experiment choice; the seed comes from the YAML training defaults.
Manifests are gitignored under `ml/data/splits/`. Existing manifests require
`--overwrite` to replace, and changed training files invalidate the split.

## Quality checks

```bash
.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts
.venv/bin/pytest backend
cd frontend && npm run lint && npm run test && npm run build
cd frontend && npx playwright install chromium && npm run test:e2e
```

## Documentation map

| Document | Purpose |
| --- | --- |
| [`docs/project-scope.md`](docs/project-scope.md) | Product and scientific scope |
| [`docs/implementation-plan.md`](docs/implementation-plan.md) | Delivery waves |
| [`docs/adding-a-model.md`](docs/adding-a-model.md) | How a later algorithm lands |
| [`docs/deploy.md`](docs/deploy.md) | Vercel + Railway |
| [`AGENTS.md`](AGENTS.md) | Multi-agent development protocol |

## License

Project code is for portfolio and academic use. MVTec AD stays under its own
non-commercial license and must not be redistributed from this repository.
