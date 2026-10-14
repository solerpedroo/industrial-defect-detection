# Industrial Defect Detection

Inspection system for **unsupervised visual anomaly detection** on industrial
images: a convolutional autoencoder and PatchCore, pixel-level localization,
and reproducible comparisons on MVTec AD.

**Stack:** Next.js (TypeScript) · FastAPI · PyTorch (from Wave 3) · Anomalib (from Wave 4)  
**Status:** Waves 1–3 data, preprocessing and offline Autoencoder implemented with synthetic CPU tests.
Real MVTec preparation and live HTTP heatmaps are pending. See [`docs/mvp-acceptance.md`](docs/mvp-acceptance.md).

## What this repository does today

- Catalogs Bottle, Leather, and Metal Nut plus the two model families
- Audits local MVTec images, mask pairs, corrupt files and duplicate content
- Creates reproducible normal-only validation manifests with source checksums
- Shows fit, validation, test and mask counts plus split validity in Datasets
- Shares recorded RGB resize/normalization across train and inference stage paths
- Provides grayscale, synchronized mask geometry and before/after preview helpers
- Trains an offline normal-only Autoencoder with Adam/MSE and early stopping
- Reloads trusted best checkpoints to produce reconstructions and error maps
- Validates inspection uploads and reports that no checkpoint is ready yet
- Dashboard pages: Overview, Datasets, Inference Lab, Experiments, Comparison

Training does not run inside the API. Measured metrics, web heatmaps, and reports
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

Preprocessing uses `ml.preprocessing.load_preprocess_config(category_id)` and
`apply_preprocess(rgb_uint8, config, stage="inference")`. The output is CHW
float32, with display RGB pixels and JSON-safe provenance in the same result.
Sizes/statistics come from YAML; all category flip permissions start disabled.
See the [preprocessing contract](docs/architecture.md#shared-preprocessing).
Offline Autoencoder consumers are implemented; PatchCore and HTTP serving remain planned.

## Offline Autoencoder

Install the training extra. For CPU tests, install the official CPU wheel first:

```bash
.venv/bin/python -m pip install "torch>=2.10,<2.11" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e "./backend[dev,training]"
.venv/bin/python scripts/train_autoencoder.py --category bottle --output-dir checkpoints/bottle-ae-001
```

Training requires a valid normal-only split manifest from the dataset validation
command. It never creates splits or opens official test images/masks. Defaults
come from YAML; overrides include `--epochs`, `--batch-size`, `--image-size`,
`--seed`, `--patience`, `--learning-rate` and `--device`. Image size must be a
multiple of eight. Existing run directories are refused. Outputs are gitignored
`best.pt` and `history.json`, containing best-validation weights and provenance.

For CUDA, install the compatible PyTorch 2.10 wheel from the
[official instructions](https://pytorch.org/get-started/previous-versions/) and
choose `--device cuda` or `cuda:0`. CUDA was not validated in the synthetic CPU run.

```python
from pathlib import Path
from ml.models.autoencoder import load_autoencoder

run = Path("checkpoints/bottle-ae-001")
predictor = load_autoencoder(run / "best.pt", trusted_root=run)
prediction = predictor.predict(rgb_uint8)
reconstruction = prediction.reconstruction  # HWC float32 [0,1], at model resolution.
error_map = prediction.anomaly_map          # HW channel-mean squared RGB error.
```

The score is the map's spatial mean. No defect decision is made before Wave 5
calibrates a threshold on validation data. The web catalog remains `planned`.

## Quality checks

The complete Python suite requires the training extra above. API-only operation
still uses the lightweight `backend[dev]` installation from Quick start.

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
