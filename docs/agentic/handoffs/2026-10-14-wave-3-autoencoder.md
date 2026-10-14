# Handoff: `wave-3-autoencoder`

- **Date:** 2026-10-14
- **From / role:** Codex / implementer and test engineer; independent id-reviewer approved
- **To / role:** id-planner / Wave 4 implementer
- **Task card:** `docs/agentic/tasks/wave-3-autoencoder.md`
- **State:** partial — engineering complete; real-MVTec, CUDA and API-serving acceptance pending

## Completed scope

Offline trainable Autoencoder with Adam/MSE, normal-only manifest consumers,
shared preprocessing, early stopping/best-state selection, loss history,
restricted checkpoint reload, square reconstructions/maps/scores and a CLI.
PyTorch is an optional training extra with CPU CI installation. No API route,
HTTP adapter, model-ready catalog change, threshold, PatchCore or frontend work.

## Exact evidence

- Commit/diff: one changed file per Wave 3 commit following baseline `298f2f5`.
  Source scope: `ml/models/autoencoder/{network,prediction,checkpoint,__init__}.py`,
  `ml/training/{config,autoencoder}.py`, `scripts/train_autoencoder.py`,
  `backend/tests/test_autoencoder.py`, `ml/configs/training_defaults.yaml`,
  `backend/pyproject.toml`, `.github/workflows/ci.yml`.
  Docs: `AGENTS.md`, `README.md`, `docs/{architecture,implementation-plan,mvp-acceptance,agent-playbook}.md`,
  `docs/agentic/{architecture-decisions,workboard}.md`, the Wave 3 card and this handoff.
- `.venv/bin/pytest backend/tests -q`: **170 passed** (121 prior + 49 Autoencoder).
- Python lint/format and diff whitespace checks: clean. Pip check: clean.
- PyTorch **2.10.0+cpu**, Python **3.14.8**; tests use one CPU thread and fixed seed 42.
- Synthetic overfit loss drops, changed patch outscores fitted constant, true
  best-epoch weights reload, same-environment runs reproduce weights/losses,
  test images/masks stay unopened, drift/failure/path-safety regressions pass.
- CLI smoke writes `best.pt` and `history.json` in pytest temporary directories
  and reloads for prediction. No real dataset or checkpoint is committed.
- Independent review approved after dtype and final-publication fixes.
- Fresh API import does not load torch; both catalog models stay `planned`.

## Decisions and conventions

ADR-010 defines BCHW float32 RGB [0,1], three trainable encoder/decoder blocks,
sigmoid output and autoencoder-specific YAML normalization (mean zero/std one).
Use `load_autoencoder_config`, `train_autoencoder` and `load_autoencoder`; the
generic future training dispatcher is not implemented. The model/predictor use
the Wave 2 pipeline. Official test pixels never enter fitting or validation.

Artifacts require a new owned directory under ignored repository roots or an
explicit external directory outside the dataset. Each normal source read verifies
the decoded byte snapshot; a final manifest check detects drift. Best weights are
independent snapshots. Completed weights are staged and hashed, history is saved,
then completed `best.pt` is published last. The checkpoint complete flag is the
readiness marker; history alone cannot prove completion after an interrupted run.

Load only `.pt` regular files under an explicit trusted root. Loading is CPU
`weights_only=True`, with schema/config/state validation before device transfer.
Incomplete checkpoints are rejected; no optimizer-state resume is provided.
Prediction is square HWC reconstruction and HW channel-mean squared RGB error
[0,1]; the score is spatial mean. No label/threshold or original-resolution map.

## Remaining risk or blocker

The licensed MVTec copy is still absent. Synthetic fitting validates engineering,
not real industrial performance or model quality. CUDA availability checks exist,
but GPU behavior/performance was not exercised with the CPU wheel. HTTP serving,
model caching, thresholds and UI heatmaps remain their later-wave responsibilities.

The autoencoder normalization differs from ImageNet feature-backbone defaults;
Wave 4 must retain its own effective config/metadata rather than reuse unit-interval
settings blindly. Maintain the same normal manifests for later comparisons.
The caller selects trusted checkpoint roots; never accept uploaded weight files.

## Next single action

Claim the existing Wave 4 card and plan the bounded Anomalib/PatchCore adapter,
checking dependency compatibility with PyTorch 2.10 and preserving the common
normal manifests. Before real-data fitting of either model, install/audit the
licensed MVTec tree and create its normal-only split manifests. Keep model catalog
status `planned` until an HTTP artifact loader and its acceptance evidence exist.
