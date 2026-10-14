# Handoff: `wave-5-evaluation`

- **Date:** 2026-10-14
- **From / role:** Codex — planner, implementer, test engineer; independent id-reviewer
- **To / role:** Wave 6 planner / inference implementer
- **Task card:** `docs/agentic/tasks/wave-5-evaluation.md`
- **State:** engineering complete and reviewed; real-MVTec experiment A remains open

## Completed scope

One offline evaluation slice: normal-validation image/pixel quantile calibration,
exact tied-score AUROC/AP, confusion/precision/recall/F1, global micro IoU/Dice,
8-connected equal-region AUPRO and an experiment A runner for both trusted model
families. All requested thresholds persist before the first official test audit
or pixel/mask read. Identical test IDs/geometry, source/artifact/history hashes,
warmed compute measurements and completed JSON/CSV publication are implemented.
No dependency, model training change, HTTP readiness, UI, study B/C or real-data
performance result was added.

## Exact evidence

- Commit/diff baseline: `5e8b820`. One changed file per commit; the Wave 5 range
  is `git log --reverse --oneline 5e8b820..HEAD` at this handoff.
- Numerical source: `ml/evaluation/{__init__,config,calibration,metrics,experiment}.py`.
- CLI/protocol: `scripts/evaluate_experiment.py`,
  `experiments/autoencoder-vs-patchcore/{config.yaml,README.md}`.
- Tests: `backend/tests/test_evaluation_{metrics,experiment}.py`.
- Documentation: `AGENTS.md`, `README.md`, `docs/{architecture,implementation-plan,
  mvp-acceptance,agent-playbook}.md`, ADR-012, Wave 5 task/workboard and this handoff.
- `PATCHCORE_PRETRAINED_SMOKE=1 .venv/bin/pytest backend/tests -q --tb=short`:
  **281 passed**, 45 upstream warnings, 46.10s. Includes the actual pretrained
  Wave 4 smoke; Wave 5 fixtures themselves explicitly use random PatchCore weights.
- `.venv/bin/pytest backend/tests -q --tb=short`: **280 passed, 1 skipped**,
  45 warnings, 45.33s. Only the optional pretrained smoke is skipped by default.
- Targeted evaluation suite: **68 passed**, 45 upstream warnings, 17.10s.
  Independent id-reviewer reran it and approved without remaining findings.
- `.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts`: pass.
  `git diff --check`: pass. `.venv/bin/python -m pip check`: no broken requirements.
- Fresh `PYTHONPATH=backend` interpreter importing `app.main` and `ml.evaluation`
  imports neither torch nor Anomalib. Model extras remain lazy/offline only.
- Python 3.14.8; torch 2.10.0+cpu; torchvision 0.25.0+cpu; Anomalib 2.7.0;
  timm 1.0.30; NumPy 2.5.4; Pillow 11.3.0. No CUDA or remote CI execution.
- Forty-seven numerical/config cases: hand AUROC .75/AP 5/6, tied/reversed/
  single-class scores, strict threshold equality, higher quantiles/empirical FPR,
  undefined null reasons, unequal-region AUPRO, clipped tie endpoints, 8-connectivity,
  lossless score conversion and 50 seed-42 comparisons with sklearn's rank metrics.
- Twenty-one integration cases: six 32×32 normal RGB images per requested
  category; seed 42, normal fraction .33, four fit/two validation IDs. Real tiny
  one-epoch Autoencoder and real Anomalib/random-backbone memory. Bottle, Leather
  and Metal Nut all run; reordered categories still have complete coverage.
- Guards cover frozen-before-audit/read ordering, identical test IDs, unchanged
  thresholds under changed test pixels/masks, incompatible split/seed/category/
  resolution/color mode, budget rejection, source/artifact/history drift, unknown
  preparation duration, interrupted final publication, CLI and existing outputs.
- Review corrections: AE duration resides in `history.json`, not its checkpoint;
  large accepted integer scores must not collapse into float64 ties; category
  coverage must not depend on presentation order. All have regression coverage.

## Decisions and conventions

- ADR-012 is authoritative; use the `evaluation-protocol` skill. It led to
  explicit all-model/all-category calibration before any test audit, fixed metric
  conventions, source provenance and synthetic-vs-real separation. The create-rule
  skill was used only to refresh the existing repository-wide AGENTS boundary.
- Canonical experiment A YAML owns target image FPR .05, pixel FPR .01, AUPRO
  FPR limit .3, warmup three and max 20,000,000 pooled pixels per category/partition.
  Overrides are explicit and hashed; the count bound is not a RAM/runtime guarantee.
- Threshold is the `higher` quantile at `1 - target_fpr`, and decisions are
  strictly `>`. Normal image scores and normal map pixels calibrate separately.
  Record observed exceedance and counts; do not claim population FPR control.
  Small image holdouts have coarse operating points; map pixels are correlated.
- AP/`aupr` is non-interpolated average precision, not trapezoidal PR area. AUROC
  groups equal scores atomically. AUPRO averages equally over 8-connected regions,
  pools background pixels, linearly clips the FPR endpoint and divides area by
  that FPR limit. Pixel IoU/Dice are global micro metrics. Undefined values are
  JSON null with reasons. Lossy float64 conversion is rejected explicitly.
- Both artifacts must match the current complete normal split, category, seed,
  resolution and inference geometry/color mode. Normalization and raw score scales
  remain family-specific. Ground truth is nearest-neighbor resized to the same
  model square; outputs do not claim original-resolution localization.
- CLI example: `scripts/evaluate_experiment.py --category bottle --data-kind
  mvtec_ad --autoencoder bottle=checkpoints/bottle-ae-001/best.pt --patchcore
  bottle=checkpoints/bottle-patchcore-001/memory.pt --output-dir runs/exp-a-001`.
  Omit category selection and provide both artifacts for all three categories
  for complete coverage. Explicit artifact parents are locally trusted roots.
  Never pass uploaded models. `--protocol` accepts an explicit YAML override.
- `data_kind` is required (`synthetic` or `mvtec_ad`), not inferred. Real experiment
  A refuses random PatchCore weights. Category coverage is partial/complete;
  resolution/protocol overrides are still visible in config/hash.
- New ignored/external runs only, outside sources. `calibration.json` has
  `complete=false` and publishes before any test audit. `results.csv` publishes
  before `results.json`, whose `complete=true` is the final report marker. Frozen
  calibration and CSV alone are not completed evaluation. Reports embed no images.
- Mean/median/p95 and per-image timings measure warmed prediction, preprocessing,
  output validation/copy; file decode/model load are excluded. CUDA synchronizes
  around timers and records allocator peak including the loaded model. CPU peak
  is approximate process-lifetime high-water RSS, not isolated model allocation.
- AE preparation time comes from regular complete `history.json` bound to weight
  checksum/config/split. Missing history means null with explicit source; invalid
  binding/duration or history drift fails. PatchCore uses checkpoint metadata.
  Neither sidecar nor metadata is a substitute for a trusted source.

## Remaining risk or blocker

- Licensed MVTec is absent. Real three-category calibration, experiment A result
  numbers and real geometry/data isolation acceptance remain open. Synthetic CPU
  evidence proves engineering paths, not industrial detection quality.
- CUDA timing/allocator paths are unverified. CPU RSS values cannot rank isolated
  model memory usage; exact pixel sorts/region walks can consume substantial RAM.
- Upstream TorchScript/Python 3.14 and timm/Anomalib import deprecation warnings
  remain visible. CI targets Python 3.12 but was not run remotely in this task.
- Both catalog models stay `planned`; HTTP inference remains 501. Wave 5 does not
  install serving/calibration loaders, overlays or download routes. Future serving
  must bind frozen thresholds to the same trusted artifact/preprocessing hashes;
  never calibrate on uploads or official test requests.
- The runner freezes order within a run but cannot prevent deliberate separate
  reruns/tuning by a user. Freeze protocol/model choices before the final test.

## Next single action

Claim and slice `wave-6-inference-ui` from its task card before changing HTTP/UI
contracts. Resolve trusted artifact/calibration loading and readiness gates in
that bounded plan, retaining the real-MVTec experiment A gate instead of marking
the offline engineering evidence as a completed real benchmark.
