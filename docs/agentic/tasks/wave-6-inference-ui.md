# Task: `wave-6-inference-ui`

- **Status:** done (engineering; real-MVTec/CUDA acceptance open)
- **Wave:** 6 — Inference lab
- **Owner / role:** Codex — planner, implementer, test engineer; independent id-reviewer
- **Model or category IDs:** `autoencoder`, `patchcore`; readiness is per installed category
- **Depends on:** Wave 5
- **Allowed files:** `backend/app/inference/`, `backend/app/visualization/`,
  `backend/app/{api/routes.py,core/config.py,main.py,schemas/catalog.py,schemas/inference.py,services/catalog.py}`,
  `backend/app/core/upload_limit.py` (bounded pre-parse multipart envelope),
  `frontend/features/inference/`, `frontend/{types/api.ts,lib/api.ts,lib/api.test.ts,playwright.config.ts}`,
  inference API/runtime/visualization tests and synthetic e2e fixtures under `backend/tests/`,
  `frontend/e2e/`, CI and README/API/architecture/plan/acceptance/playbook/ADR/task/workboard/handoff/AGENTS docs.

Planner clarification (baseline `9826889`): one bounded upload-to-result vertical
slice. No dependency or new endpoint. An optional operator-owned serving manifest
binds trusted checkpoints to Wave 5 frozen `calibration.json`; validate hashes,
configs, split provenance and quantile convention at startup. Preload/cache/freeze
predictors and serialize use; request handlers never train/calibrate/read datasets.
Default unconfigured API remains lightweight/planned/501. Runtime catalog readiness
adds category coverage without changing canonical model IDs or claiming real-data
acceptance. Synthetic serving needs explicit opt-in and visible warnings. Render
at recorded model resolution with threshold-fixed heat colors, strict binary mask,
optional reconstruction and ephemeral base64 PNG downloads. Compare transport may
reuse the two existing predictors; saved-experiment comparison UI stays Wave 7.
Browser CI uses tiny synthetic real-model fixtures; one changed file per commit.

## Goal

Let a user upload one image, pick a category and a ready model, and see the
score, binary decision, heatmap, overlay with opacity, and a mask download.

## Contract and conventions

ADR-005 and ADR-006. Bounding boxes are optional contours of the mask.
`inference_ms` excludes decode and PNG encode.

## Plan and acceptance criteria

- [x] `501` remains for uninstalled/planned category/model pairs.
- [x] Ready models return the JSON contract in `docs/api-contract.md`.
- [x] UI states API errors and does not keep a stale map after a failed request.
- [x] Startup rejects invalid, escaping, mismatched or synthetic-unapproved artifacts/calibration.
- [x] Cached inference never trains, recalibrates, reloads per request or persists uploaded payloads.
- [x] Heatmap, opacity overlay, strict pixel mask and PNG download use model-resolution geometry.
- [x] Browser exercises real API/model results, selection invalidation, failure cleanup and download.

## Tests and commands

```bash
.venv/bin/pytest backend
cd frontend && npm run test && npm run test:e2e
cd frontend && npm run lint && npm run build
.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts
```

## Non-goals

No multi-experiment dashboard (Wave 7).

## Evidence / handoff

- Full suite with actual pretrained Wave 4 smoke: **319 passed**, 45 retained
  upstream warnings, 64.06s. New runtime/visualization/transport suite:
  **38 passed**, 45 warnings, 21.17s. Independent id-reviewer repeated it and
  approved with no remaining findings.
- Default `.venv/bin/pytest backend -q --tb=short`: **318 passed, 1 skipped**,
  45 warnings, 62.92s; only the optional pretrained smoke is skipped.
- Frontend ESLint, four API-client unit tests and production build pass.
- Real-model browser suite: **3 passed**, 6.3s, using isolated API 8001/web 3001.
  Build used `NEXT_PUBLIC_API_URL=http://127.0.0.1:8001`; fixture CORS used the
  matching web origin. Existing services on 8000/3000 were not interrupted.
- Browser evidence covers actual HTTP200/model-square PNGs, synthetic warning,
  opacity, downloaded PNG signature, both model families, selection/error cleanup,
  a delayed obsolete success and all dashboard routes. A generated screenshot
  was visually inspected; no dataset/model/image payload is committed.
- Review findings fixed with regressions: move Pillow decode off async event
  loop; lazy-import model families so AE-only serving never imports Anomalib.
- Tests also cover normal source/config/threshold provenance failures, FIFO/
  symlink/escape/schema rejection, opt-in, cached source-free operation, startup
  readiness, planned/4xx/503 responses, strict mask equality, zero threshold,
  fixed color saturation, reconstruction and bounded chunked/declared envelopes.
- Synthetic fixture: Bottle, six 32×32 distinct normal RGB images, seed42,
  fraction .33 (four fit/two validation), tiny real one-epoch Autoencoder and
  real Anomalib/random backbone. Serving never runs this training fixture;
  Playwright's separate test bootstrap prepares it before starting the API.
- Unconfigured API/registry import loads no torch/torchvision/Anomalib/timm.
  Pip check reports no broken requirements. No dependency or endpoint was added.
- Upstream Python3.14/TorchScript/import warnings remain visible; real licensed
  MVTec quality, CUDA and remote CI were not verified. Canonical defaults remain
  planned; synthetic readiness is not industrial acceptance.
- One changed file per commit; non-obvious continuation context:
  `docs/agentic/handoffs/2026-10-14-wave-6-inference-ui.md`.
