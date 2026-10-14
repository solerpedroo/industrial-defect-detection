# Industrial Defect Detection

Inspection system for **unsupervised visual anomaly detection** on industrial
images: a convolutional autoencoder and PatchCore, pixel-level localization,
and reproducible comparisons on MVTec AD.

**Stack:** Next.js (TypeScript) · FastAPI · PyTorch (from Wave 3) · Anomalib (from Wave 4)  
**Status:** Waves 1–4 data, preprocessing, offline Autoencoder and PatchCore implemented with synthetic CPU tests.
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
- Builds frozen pretrained ResNet-18 PatchCore coresets from normal fit images
- Reloads self-contained PatchCore artifacts offline to produce raw scores/maps
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
Both offline model consumers are implemented; HTTP serving remains planned.

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

## Offline PatchCore

Install compatible CPU wheels first, then the optional model packages:

```bash
.venv/bin/python -m pip install "torch>=2.10,<2.11" "torchvision>=0.25,<0.26" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e "./backend[dev,training,patchcore]"
.venv/bin/python scripts/build_patchcore.py --category bottle --output-dir checkpoints/bottle-patchcore-001
```

Anomalib 2.7.0 supplies the low-level PatchCore model, not its trainer or automatic
calibration. The first pretrained run retrieves public `timm/resnet18.a1_in1k`
weights into the local cache. The backbone remains frozen/eval; only normal fit
images populate memory. Manifest validation hashes normal held-out files but
never decodes their pixels. Official test images and masks are not opened.

ImageNet RGB normalization and defaults come from YAML. Overrides include
`--image-size` (multiple of 32), `--batch-size`, `--seed`, `--coreset-ratio`,
`--num-neighbors`, `--max-patches` and `--device`. Too many patches or too few
coreset neighbors fail explicitly; the builder never silently downsamples.
The patch budget bounds count, not guaranteed RAM/VRAM or runtime. K-center
sampling can be expensive at full resolution. `--no-pretrained` is explicit
random-weight regression mode, not an industrial pretrained experiment.

Each new ignored/external run writes `metadata.json` and publishes completed
`memory.pt` last. It records config/source hashes, seed, versions, frozen backbone
digest, memory dimensions and hardware. Reload includes all backbone weights and
does not retrieve pretrained weights again:

```python
from pathlib import Path
from ml.models.patchcore import load_patchcore

run = Path("checkpoints/bottle-patchcore-001")
predictor = load_patchcore(run / "memory.pt", trusted_root=run)
prediction = predictor.predict(rgb_uint8)
score = prediction.score             # Raw weighted maximum-patch distance.
anomaly_map = prediction.anomaly_map  # Square HW float32 at model resolution.
```

Both predictors implement `ml.models.AnomalyPredictor` and return an
`AnomalyPrediction`. PatchCore distances are unbounded and are not numerically
comparable to Autoencoder MSE. There is no calibrated threshold or defect label.
Only load locally trusted artifacts, never uploaded checkpoints. Synthetic CPU
tests include a real pretrained smoke, not real-MVTec or CUDA acceptance.
Upstream TorchScript/Python 3.14 and deprecated-import warnings remain visible.

## Quality checks

The complete Python suite requires both `training` and `patchcore` extras above. API-only operation
still uses the lightweight `backend[dev]` installation from Quick start.

```bash
.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts
.venv/bin/pytest backend
# Optional real pretrained-weight smoke (initial run may download public weights):
PATCHCORE_PRETRAINED_SMOKE=1 .venv/bin/pytest backend/tests/test_patchcore.py -q
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
