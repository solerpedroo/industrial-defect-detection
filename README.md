# Industrial Defect Detection

Inspection system for **unsupervised visual anomaly detection** on industrial
images: a convolutional autoencoder and PatchCore, pixel-level localization,
and reproducible comparisons on MVTec AD.

**Stack:** Next.js (TypeScript) · FastAPI · PyTorch (from Wave 3) · Anomalib (from Wave 4)  
**Status:** Waves 1–7 data, offline models/evaluation, calibrated HTTP inspection and comparison implemented with synthetic CPU tests.
Real MVTec preparation, benchmark and CUDA acceptance remain pending. See [`docs/mvp-acceptance.md`](docs/mvp-acceptance.md).

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
- Freezes normal-validation thresholds before any official test evaluation
- Exports image/pixel metrics and documented compute measurements as JSON/CSV
- Serves installed category/model pairs with frozen thresholds and ephemeral PNG results
- Shows heatmaps, adjustable overlays, reconstruction and downloadable masks
- Compares two installed models on one upload and opens saved experiment metrics locally
- Filters saved comparisons by category and exports their values/provenance as CSV
- Dashboard pages: Overview, Datasets, Inference Lab, Experiments, Comparison

Training does not run inside the API. Offline experiment A is runnable; real-MVTec
results and presentation reports remain pending. Protocols A, B, and
C are specified under [`experiments/`](experiments/README.md).

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
Both model families support offline and explicitly configured HTTP consumers.

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

The score is the map's spatial mean. Bare prediction makes no defect decision;
offline evaluation below calibrates a threshold on normal validation data.
The default web catalog remains `planned` until compatible serving artifacts are installed.

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
comparable to Autoencoder MSE. Bare prediction attaches no threshold or defect label.
Only load locally trusted artifacts, never uploaded checkpoints. Synthetic CPU
tests include a real pretrained smoke, not real-MVTec or CUDA acceptance.
Upstream TorchScript/Python 3.14 and deprecated-import warnings remain visible.

## Offline evaluation — experiment A

First prepare audited normal split manifests and both completed model artifacts.
All requested categories/models calibrate on held-out normals before any test
audit or decoding. Defaults live in `experiments/autoencoder-vs-patchcore/config.yaml`:
image FPR target 0.05, pixel FPR target 0.01, higher quantiles and strict `>`
decisions. These are empirical calibration targets, not guaranteed population
false-positive rates. Small validation sets give coarse operating points.

For a single-category partial run:

```bash
.venv/bin/python scripts/evaluate_experiment.py --category bottle --data-kind mvtec_ad \
  --autoencoder bottle=checkpoints/bottle-ae-001/best.pt \
  --patchcore bottle=checkpoints/bottle-patchcore-001/memory.pt \
  --output-dir runs/experiment-a-bottle-001
```

Omit `--category` and repeat both artifact arguments for Leather and Metal Nut
to cover the full three-category protocol. Each explicitly supplied checkpoint
directory is locally trusted; never use uploaded checkpoints. `--data-kind`
declares `mvtec_ad` or `synthetic`. Real experiment A refuses random PatchCore
weights. `--protocol` accepts an explicit experiment A YAML override; changes
are hashed/recorded and must be fixed before reporting on official test data.
`--dataset-root`, `--manifest-dir` and `--device` support local alternatives.

Artifacts must match category, normal split/checksums, experiment seed, resolution
and inference geometry. Test IDs are shared; masks resize with nearest-neighbor
to model resolution. Exact tied-score AUROC, non-interpolated AP (`aupr`), image
confusion/precision/recall/F1, global pixel IoU/Dice and 8-connected AUPRO are
reported. Undefined metrics are `null` with reasons, not fake zeros.

A new ignored/external run writes `calibration.json` before test reads, then
`results.csv` and completed `results.json` last. CSV alone is not completion.
Reports include raw image scores, thresholds, source/artifact hashes, config,
versions and hardware, without image/map payloads. Warmed prediction timing
includes preprocessing/output copy, excludes file decode/model load; CUDA is
synchronized. CPU peak memory is process-lifetime high-water RSS, not isolated
model allocation. Autoencoder prepare time uses checkpoint-bound `history.json`;
when absent it is explicitly unknown. Pixel budgets fail rather than subsample.

Only synthetic CPU experiment evidence is available today. The real licensed
MVTec experiment and CUDA timing remain acceptance gates. See
[`experiment A`](experiments/autoencoder-vs-patchcore/README.md).

## Calibrated HTTP inference

Install the corresponding model extra (`training` for Autoencoder, `patchcore`
for PatchCore) and prepare compatible checkpoints plus Wave 5 `calibration.json`.
No model is enabled merely because a `.pt` file exists. Create an operator-owned
manifest outside tracked source files, for example `runs/serving.json`:

```json
{
  "schema_version": 1,
  "trusted_root": "/absolute/path/to/industrial-defect-detection",
  "device": "cpu",
  "entries": [
    {
      "category_id": "bottle",
      "model_id": "autoencoder",
      "checkpoint": "checkpoints/bottle-ae-001/best.pt",
      "calibration": "runs/experiment-a-bottle-001/calibration.json"
    },
    {
      "category_id": "bottle",
      "model_id": "patchcore",
      "checkpoint": "checkpoints/bottle-patchcore-001/memory.pt",
      "calibration": "runs/experiment-a-bottle-001/calibration.json"
    }
  ]
}
```

All entry paths are canonical relative paths under the explicit trusted root;
symlinks, escapes and special files are rejected. Never install uploaded models
or untrusted JSON. Start the API with server-only configuration:

```bash
INFERENCE_MANIFEST=/absolute/path/to/runs/serving.json \
  .venv/bin/uvicorn app.main:app --app-dir backend --port 8000
```

Startup validates checkpoint/calibration hashes, configuration, preprocessing,
normal split membership/checksums and higher-quantile threshold provenance. It
loads, freezes and warms predictors once, with no dataset files required. A
frozen calibration is usable independently of a completed test report. Invalid
installations fail startup; restart to change the installed snapshot. Prediction
never trains, reloads weights or chooses thresholds from uploads.

`GET /api/models` exposes runtime `ready_categories` and `synthetic_categories`.
Uninstalled pairs retain `501`, even when another category makes the model
globally `ready`. Without a manifest the API remains lightweight and `planned`.
Synthetic calibration additionally requires `INFERENCE_ALLOW_SYNTHETIC=1` and
produces a visible demonstration warning; it is not industrial acceptance.

Inference Lab shows prepared input, fixed-scale heatmap, strict binary pixel mask,
optional reconstruction and an opacity-adjustable overlay. Maps/downloads share
model resolution, not original coordinates. Scores remain family-specific, not
severity/probability; image and pixel decisions use separate normal-validation
thresholds with strict `>`. Changing inputs or failed/obsolete requests clears
old results. Heat colors use [0,2×pixel threshold], not upload min/max.

Uploads/results are not persisted by the application. Multipart bodies are bounded
before parsing (10 MiB plus 64 KiB overhead); temporary framework spools close at
request end. Prediction/decoding/PNG work runs off the async event loop. Latency
excludes decode/PNG encode and includes model preprocessing; CUDA is synchronized
when selected, but remains unverified. `ALLOWED_ORIGINS` must include your UI
origin if it differs from the default localhost ports. Do not expose serving
configuration through `NEXT_PUBLIC_*` variables.

## Compare models and saved experiments

Open **Comparison** and select two distinct models installed for the same category.
One upload produces two heatmaps, scores, frozen thresholds, inference timings
and downloadable masks. The overlay slider controls both images. Each heatmap
retains its own calibration scale; raw scores/colors do not rank model quality.
Changing the upload or selectors clears previous results, including delayed calls.

In **Saved experiment metrics**, select the completed `results.json` produced by
the offline evaluation command above. The reader accepts up to 16 MiB, checks
schema/completion, catalog membership, row coverage, test IDs/labels, numbers and
calibration consistency, and keeps the report in browser memory. It does not
upload the file, authenticate its provenance or recalculate metrics. Select a
saved category and **Export visible table CSV** to download those recorded values
with hashes, seed, thresholds and compute context. Undefined metrics stay blank
in CSV with their reason, and appear as N/A in the table.

Saved test metrics are independent of the current upload and installed model
snapshot. Synthetic/partial reports remain explicitly labeled. Timing excludes
file decoding/model loading; CPU peak memory is a process high-water mark, not
isolated model usage. A report larger than the explicit limit requires a later
pagination/serving design, not silent truncation. No reports endpoint was added.

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

Browser tests start an isolated tiny synthetic real-model API fixture (both model
extras required), not a mock predictor. No pretrained download is needed. Use
`PLAYWRIGHT_REAL_INFERENCE=0` only for a shell-only lightweight check; inference
and comparison flows are then skipped. The fixture also supplies a completed
synthetic evaluation JSON under ignored `frontend/.next/` for local-file tests.
To preserve already running servers, choose free ports:

```bash
cd frontend
NEXT_PUBLIC_API_URL=http://127.0.0.1:8001 npm run build
PLAYWRIGHT_API_URL=http://127.0.0.1:8001 PLAYWRIGHT_WEB_URL=http://localhost:3001 \
  PLAYWRIGHT_PRODUCTION=1 npm run test:e2e
```

Public API URLs are baked into a production build, not changed by `next start`.

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
