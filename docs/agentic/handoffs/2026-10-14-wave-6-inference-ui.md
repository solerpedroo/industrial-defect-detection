# Handoff: `wave-6-inference-ui`

- **Date:** 2026-10-14
- **From / role:** Codex — planner, implementer, test engineer; independent id-reviewer
- **To / role:** Wave 7 planner / comparison implementer
- **Task card:** `docs/agentic/tasks/wave-6-inference-ui.md`
- **State:** engineering complete and reviewed; real-MVTec/CUDA acceptance open

## Completed scope

One calibrated upload-to-result vertical slice. Startup-only operator-owned
serving manifest binds trusted model weights to frozen Wave 5 calibration;
predictors are loaded/frozen/warmed/cached and used serially. Existing inference
and compare transport serve ephemeral PNGs with strict image/pixel decisions.
Inference Lab displays prepared input, fixed-calibration heatmap, mask, optional
reconstruction and adjustable overlay; downloads and stale/error cleanup are
implemented. No new dependency/endpoint, training feature, public artifact upload,
saved-experiment dashboard or real-data performance claim was added.

## Exact evidence

- Commit/diff baseline: `9826889`; one changed file per commit. At this handoff,
  `git log --reverse --oneline 9826889..HEAD` recovers the Wave 6 range.
- Backend source: `backend/app/inference/{__init__,runtime}.py`,
  `backend/app/visualization/{__init__,render}.py`, `backend/app/api/routes.py`,
  `backend/app/core/{config,upload_limit}.py`, `backend/app/main.py`,
  `backend/app/schemas/{catalog,inference}.py`, `backend/app/services/catalog.py`.
- Frontend source: `frontend/features/inference/{inference-lab,result-view}.tsx`,
  `frontend/types/api.ts`, `frontend/lib/api.ts`.
- Tests/CI: `backend/tests/{inference_fixture,test_inference_runtime}.py`,
  `frontend/lib/api.test.ts`, `frontend/e2e/inference.spec.ts`,
  `frontend/playwright.config.ts`, `.github/workflows/ci.yml`.
- Documentation: AGENTS, README, API contract, architecture, implementation plan,
  MVP acceptance, playbook, ADR-013, Wave 6 task/workboard and this handoff.
- `PATCHCORE_PRETRAINED_SMOKE=1 .venv/bin/pytest backend/tests -q --tb=short`:
  **319 passed**, 45 retained upstream warnings, 64.06s. Includes the real
  pretrained Wave 4 smoke, not a real-MVTec experiment.
- `.venv/bin/pytest backend -q --tb=short`: **318 passed, 1 skipped**,
  45 warnings, 62.92s; default skips only the optional pretrained smoke.
- Targeted runtime suite: **38 passed**, 45 warnings, 21.17s. Independent reviewer
  reran it, lint and whitespace checks: approved, no remaining findings.
- `cd frontend && npm run lint && npm run test`: pass, **4 client tests**.
- `cd frontend && NEXT_PUBLIC_API_URL=http://127.0.0.1:8001 npm run build`: pass.
- `PLAYWRIGHT_API_URL=http://127.0.0.1:8001 PLAYWRIGHT_WEB_URL=http://localhost:3001
  PLAYWRIGHT_PRODUCTION=1 npm run test:e2e -- --max-failures=1` from frontend:
  **3 passed**, 6.3s. Actual API/model predictions, not a mocked predictor.
- Browser story: upload 21×35 RGB PNG -> installed Bottle Autoencoder/PatchCore
  -> HTTP200 with 32×32 maps -> visible warning/score/decision -> 75% opacity ->
  downloaded valid PNG mask -> selection and controlled failure remove results ->
  obsolete delayed success cannot restore them. Dashboard routes also pass.
- A generated full-page screenshot was visually inspected; no screenshot or
  image/checkpoint/calibration payload enters Git. Existing 8000/3000 services
  were preserved; test servers on 8001/3001 stopped after the run.
- `.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts`: pass.
  Pip check: no broken requirements. Fresh unconfigured app/registry imports
  leave torch/torchvision/Anomalib/timm absent from `sys.modules`.
- Synthetic fixture: six distinct Bottle 32×32 normal RGB images, seed42,
  fraction .33, four fit/two validation; one-epoch tiny real Autoencoder and
  Anomalib with explicit random ResNet weights. Separate test bootstrap prepares
  artifacts before starting the API; no HTTP request runs this fixture/training.
- Coverage: startup schema/hash/config/preprocessing/normal split/threshold
  provenance, opt-in, symlink/FIFO/path escapes, safe readiness/4xx/501/503,
  source-free cached snapshots, no request weight reload/calibration/dataset read,
  off-loop decode/closed uploads, bounded declared/chunked envelopes, fixed color
  range, threshold equality/zero case, reconstruction and PNG geometry/content.
- Review findings corrected with regressions: Pillow decode must leave the async
  event loop; AE-only serving must not import the optional PatchCore stack.
- Browser setup corrections: production public API URL is fixed at build time;
  fixture CORS must match the alternate web origin; use accessible combobox
  selectors for controls whose wrapping labels include options.

## Decisions and conventions

- ADR-013 is authoritative. No canonical category/model IDs or numeric defaults
  are duplicated in the UI. YAML models remain planned by default; runtime
  readiness comes from validated installed category/model pairs.
- `INFERENCE_MANIFEST` points to local JSON with schema_version1, absolute
  `trusted_root`, `device` and `entries` (category_id/model_id/relative checkpoint/
  relative calibration). See README for the complete shape. Never accept serving
  JSON or checkpoints from an upload. Paths must be regular, non-symlink and
  canonical under the explicit root. Invalid installation fails startup.
- Use Wave 5 `calibration.json` with `complete=false` and phase
  `thresholds_frozen_before_test`, not a test-selected threshold or completed
  results JSON. Calibration can serve independently of successful test reporting.
  Startup checks weight/config/preprocessing hashes, normal source split and
  deterministic validation membership/checksums, quantile/method/comparator and
  image/pixel sample counts. No dataset files are opened by serving.
- Synthetic calibration additionally requires `INFERENCE_ALLOW_SYNTHETIC=1`;
  response `data_kind` and UI warning distinguish demonstration from real data.
  This explicit opt-in never closes industrial quality acceptance.
- `GET /api/models` returns `ready_categories` and `synthetic_categories`; global
  ready status means at least one pair, not all categories. Uninstalled pairs
  still return 501. Request input IDs are catalog-validated without dataset reads.
- Lifespan preloads/freezes/warms family-specific predictors once. AE-only needs
  the `training` extra; PatchCore needs `patchcore`. No configured manifest means
  no optional model imports. Restart to install changed weights/calibration;
  loaded snapshots are intentionally not rehashed/reloaded per request.
- Decode and prediction/PNG work run in the thread pool; predictor use is
  serialized per process. `inference_ms` synchronizes CUDA around prediction/
  preprocessing and excludes decode/encode/queue waiting. `request_ms` includes
  decode/queued work/rendering before JSON serialization/network transfer.
- ASGI envelope bounds inference multipart bodies to 10 MiB+64KiB before parsing,
  including chunked streams. File reads also cap at limit+1 and close on all
  handled paths. Framework temporary spools may use disk, but no application
  upload/result persistence or payload logging is added. Transport errors are
  explicit; model failure is sanitized 503 without host details.
- Strict image decision is score > image threshold; mask is raw map > pixel
  threshold (equality normal). Heat uses [0,2×pixel threshold], not upload min/max;
  threshold zero uses zero/nonzero intensity. No physical severity or cross-model
  probability interpretation. All PNGs share model-square resolution.
- Response includes input/heat/50% overlay/mask/optional reconstruction PNG,
  raw image score, both thresholds, dimensions, data kind and identifying hashes.
  UI opacity is client-only; its download is clearly labeled 50% overlay.
- UI actions submit explicitly. Any input change, resubmission or failure clears
  previous results/downloads. AbortController plus generation guards suppress
  obsolete success/failure; unmount cancels outstanding work.
- Model/evaluation skills enforced calibration binding and no HTTP fitting. React
  checklist shaped derived state, transient request refs and accessible controls;
  full-story verification found build-time URL/CORS test setup issues and verified
  all boundaries. Backend guidance preserved FastAPI deployment. create-rule was
  used only to refresh existing repository-wide AGENTS status, not add new rules.
- E2E CI now installs CPU wheels and both model extras for the real-model fixture.
  `PLAYWRIGHT_REAL_INFERENCE=0` is explicitly shell-only and skips inference flows.
  For alternate-port production tests, set NEXT_PUBLIC_API_URL during build;
  setting it only for next start cannot change the browser bundle.

## Remaining risk or blocker

- Licensed MVTec preparation, real experiment A and actual industrial detection
  quality remain open. Synthetic HTTP200/PNG correctness is engineering evidence,
  not acceptance of a trained industrial inspection model.
- CUDA serving/synchronization is implemented but unverified; no GPU latency,
  VRAM capacity or deployment performance claim is supported. Multiple workers
  each load their own model copies. Model-resolution square stretching is not
  original-coordinate localization.
- Python3.14/TorchScript and timm/Anomalib deprecation warnings remain visible.
  Local tests use Python3.14.8/torch2.10.0+cpu/torchvision0.25.0+cpu/Anomalib2.7.0/
  timm1.0.30/NumPy2.5.4/Pillow11.3.0; remote CI was not executed by this task.
- Serving remains an operator-controlled local/deployment installation, without
  authentication, rate limiting or global concurrency quotas. Those deployment
  hardening controls are not broadened into this wave; do not expose an untrusted
  public installation without its deployment safety review.
- Saved-experiment dashboards, measured comparisons and report export remain
  Waves 7/8. Existing compare transport is ephemeral and should not be mistaken
  for completed real experiment A comparison or re-used to tune test thresholds.

## Next single action

Claim/slice `wave-7-comparison` from its card. Consume trusted saved experiment
artifacts with config/source provenance, keep synthetic/real evidence separate
and do not turn catalog defaults or synthetic browser fixtures into fabricated
benchmark tables. Preserve the still-open real-MVTec and CUDA gates.
