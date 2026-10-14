# Handoff: `wave-4-patchcore`

- **Date:** 2026-10-14
- **From / role:** Codex — planner, implementer, test engineer; independent id-reviewer
- **To / role:** Wave 5 planner / evaluation implementer
- **Task card:** `docs/agentic/tasks/wave-4-patchcore.md`
- **State:** engineering complete and reviewed; real-data/CUDA/HTTP gates open

## Completed scope

Frozen pretrained ResNet-18 PatchCore through Anomalib's low-level model;
normal-fit-only feature memory, seeded bounded k-center coreset, recorded shared
preprocessing, offline CLI, self-contained trusted artifacts and score/map
prediction. A common protocol/DTO spans Autoencoder and PatchCore without implying
the same numerical score scale. No calibration, metrics, HTTP serving, frontend
feature, model readiness or benchmark was added.

## Exact evidence

- Commit/diff: Wave 4 starts after `28f6208`; exactly one changed file per commit.
  Recover the range with `git log --reverse --oneline 28f6208..HEAD` at this handoff.
  No dataset image, mask, pretrained cache or generated artifact is committed.
- Source scope: `ml/models/patchcore/{__init__,config,adapter,artifact,build}.py`,
  `ml/models/__init__.py`, `scripts/build_patchcore.py`,
  `backend/tests/test_patchcore.py`, `ml/configs/training_defaults.yaml`,
  `backend/pyproject.toml`, `.github/workflows/ci.yml`.
- Documentation scope: `AGENTS.md`, `README.md`, `docs/architecture.md`,
  `docs/implementation-plan.md`, `docs/mvp-acceptance.md`, `docs/agent-playbook.md`,
  `docs/agentic/architecture-decisions.md`, Wave 4 task/workboard and this handoff.
- `PATCHCORE_PRETRAINED_SMOKE=1 .venv/bin/pytest backend/tests -q`:
  **213 passed**, 45 upstream warnings in 36.66s. All 43 PatchCore cases passed.
- `.venv/bin/pytest backend/tests -q`: **212 passed, 1 skipped**, 45 warnings
  in 35.82s. Default tests skip only the opt-in pretrained smoke.
- `.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts`: pass.
- `.venv/bin/python -m pip check`: no broken requirements.
- Fresh `PYTHONPATH=backend` interpreter importing `app.main`: neither torch nor
  Anomalib appears in `sys.modules`; API-only import remains lightweight.
- Independent id-reviewer reran all 43 PatchCore tests with pretrained smoke,
  lint and whitespace checks: approved with no concrete findings.
- Python 3.14.8; torch 2.10.0+cpu; torchvision 0.25.0+cpu; Anomalib 2.7.0;
  timm 1.0.30; NumPy 2.5.4; Pillow 11.3.0. CPU-only execution.
- Synthetic fixture: six 32×32 normal images; seed 42 and validation fraction
  0.33 leave four fit IDs. Layer2/layer3 produce 64 patches, ratio 0.5 retains
  32 rows of dimension 384. Bright-patch corruption scores above a normal image.
- Coverage: guarded held-out/test/mask decoding, exact-byte checksums and drift,
  frozen weights/BatchNorm, deterministic same-environment builds, scoped/restored
  RNG/dtype/backend state, memory budget/minimum neighbors, shape/raw score
  contract, safe restricted offline reload, malformed state/schema, symlink/FIFO
  rejection, CLI failure paths and staged-last publication failures.
- Most regression cases explicitly use random weights to avoid a network
  dependency. The separately enabled smoke fetched actual public
  `timm/resnet18.a1_in1k` weights, built on synthetic normals, then reloaded with
  pretrained retrieval forbidden. This is not real-MVTec performance evidence.

## Decisions and conventions

- ADR-011 is authoritative. Optional `patchcore` extra pins Anomalib 2.7.0,
  torchvision 0.25.x and timm 1.0.x with torch 2.10.x. Backend CI installs CPU
  wheels and both model extras; default smoke never requires pretrained retrieval.
- Build after `scripts/validate_dataset.py --create-split`; use a new directory
  under ignored `checkpoints/`, `artifacts/`, `runs/` or outside the repository,
  never inside the source dataset. CLI: `scripts/build_patchcore.py --category
  bottle --output-dir checkpoints/bottle-patchcore-001`. Default is pretrained;
  `--no-pretrained` is recorded regression mode only.
- Normal fit images alone populate memory. Manifest checks hash normal validation
  bytes without decoding their pixels. Official test images/masks are never
  opened during building. Train and prediction share ImageNet-normalized RGB;
  no fit augmentation is applied. Autoencoder retains unit-interval RGB/MSE.
- Coreset count is `floor(patches * ratio)` and must be at least `num_neighbors`.
  Explicit patch budget rejects oversized fits rather than changing sample
  selection/ratio. K-center remains potentially expensive; budget is not a
  guarantee of memory capacity or wall-clock speed.
- `memory.pt` contains backbone weights, memory and primitive provenance.
  `metadata.json` adds its exact artifact checksum. JSON publishes first, completed
  checkpoint last; an unsuccessful run directory can remain for diagnosis.
  Restricted CPU `weights_only=True` loading still requires an explicit locally
  trusted root. Do not accept model artifacts from uploads.
- `AnomalyPredictor.predict(rgb_uint8)` returns `AnomalyPrediction` with finite
  nonnegative score and square HW float32 map. PatchCore score is the raw weighted
  maximum-patch nearest-neighbor score (plain max for one neighbor), not map mean;
  its smoothed map is not guaranteed to share the score's maximum. Distances have
  no unit upper bound and cannot be compared directly to Autoencoder MSE.
- Existing atomic JSON/checkpoint write helpers and device validation are reused
  from the Autoencoder package; no Autoencoder source implementation was changed.
- AGENTS changes refresh the repository-wide boundary only, following the
  create-rule skill's focused persistent-guidance scope; no new rule was added.

## Remaining risk or blocker

- Licensed MVTec Bottle/Leather/Metal Nut are absent. Audit and normal-only split
  preparation remain prerequisites to real experiments. No industrial accuracy,
  calibrated threshold, benchmark timing, VRAM or CUDA claim is supported.
- The tested Python 3.14 environment retains upstream TorchScript compatibility
  warnings plus timm/Anomalib deprecated-import warnings. No global suppression
  was added. CI targets Python 3.12; remote CI was not executed by this task.
- Both catalogs remain `planned`; HTTP inference remains 501. Offline loading
  is not serving acceptance. Thresholds, metrics and labels belong to Wave 5;
  heatmaps/UI/HTTP serving belong to their later bounded tasks.
- Bit-identical results across hardware/library versions are not guaranteed.
  Pretrained first build needs cache/network. Subsequent artifact reload does not.

## Next single action

Claim `wave-5-evaluation` and review its task card before code. Define metrics and
normal-validation false-positive calibration without selecting anything on
official test images; keep synthetic engineering evidence separate from the
still-open real-MVTec experiment A acceptance gate.
