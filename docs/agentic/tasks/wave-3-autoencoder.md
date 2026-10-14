# Task: `wave-3-autoencoder`

- **Status:** done — engineering; real-MVTec, CUDA and API-serving acceptance pending
- **Wave:** 3 — Convolutional autoencoder
- **Owner / role:** Codex — planner, implementer, test engineer; independent id-reviewer
- **Model or category IDs:** `autoencoder`
- **Depends on:** Wave 2, ADR that adds PyTorch
- **Allowed files:** `ml/models/autoencoder/`, `ml/training/`, `scripts/train_autoencoder.py`,
  `ml/configs/training_defaults.yaml`, `backend/pyproject.toml`, `.github/workflows/ci.yml`,
  `backend/tests/test_autoencoder.py`, ADR/task/workboard/handoff records and
  architecture/README/implementation/acceptance/playbook/agent-boundary docs.

Planner scope: add the optional PyTorch training extra and CPU CI installation;
keep the web API lightweight and the catalog `planned`. No API route or adapter
is added. The user authorized Wave 3 engineering with synthetic evidence while
the licensed real-MVTec gate remains open. Commit each changed file separately.

## Goal

Train a convolutional autoencoder on normal images with Adam, MSE, early
stopping on the normal validation split, and a saved checkpoint. Inference
returns a reconstruction and an error map.

## Contract and conventions

Input tensors are documented (size, normalization). The error map is a
non-negative float array. A high reconstruction error is not automatically a
defect label until Wave 5 supplies a threshold.

ADR-010 records PyTorch 2.10, normal-only manifest consumers, unit-interval RGB
normalization, the encoder/decoder, best-validation checkpoint publication,
restricted trusted-root loading, reproducibility and artifact provenance.

## Plan and acceptance criteria

- [x] Encoder and decoder restore spatial size.
- [x] Training CLI writes a checkpoint outside Git and a loss history JSON.
- [x] Synthetic overfitting test: loss falls on a constant image batch.
- [x] Catalog status stays `planned` until the API can load the checkpoint.
- [x] Train and prediction use the recorded Wave 2 pipeline; no test pixels enter fit/selection.
- [x] Dataset drift, invalid configurations and unsafe checkpoint paths fail explicitly.
- [x] Inference uses eval/inference_mode and returns a finite map/score without a label.
- [x] Checkpoint reload reproduces predictions; history records config, split, seed and versions.
- [x] Existing artifacts are not overwritten; early stopping saves the best epoch, not the last.

## Tests and commands

```bash
.venv/bin/pytest backend/tests -q
.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts
```

## Non-goals

No PatchCore, no pixel AUROC, no UI heatmap yet.
No real-dataset download, API inference implementation, binary decision or automatic
claim of real industrial detection accuracy. Resume training is not implemented.

## Evidence / handoff

- `.venv/bin/python -m pip install 'torch>=2.10,<2.11' --index-url https://download.pytorch.org/whl/cpu`:
  installed **2.10.0+cpu**, Python **3.14.8**. Editable `backend[dev,training]` install passed.
- `.venv/bin/pytest backend/tests -q`: **170 passed**, including **49 Autoencoder cases**.
- `.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts`: **clean**.
- `.venv/bin/python -m pip check`: **No broken requirements found**.
- Fresh-process API import verified `torch` absent from `sys.modules`; the optional
  training extra does not alter the API import path.
- Synthetic RGB/masks: 8×8 uint8 [0,255], seed 42, six distinct normal training
  sources and a manifest holding out two for validation. Shape regressions cover
  8, 16 and canonical 256 pixels. No licensed images or weights are committed.
- Constant RGB 0.2 batch: 30 Adam steps reduce MSE below 20% of the initial loss;
  a changed bright patch outscores the fitted constant. This is a toy numerical
  regression, not evidence of real industrial classification quality.
- Hand-computed channel-MSE map and weighted partial-batch loss pass. Controlled
  validation losses save epoch-2 weights rather than epoch 4, with patience stopping.
- Two same-environment CPU runs reproduce losses and weights. Caller RNG/backend
  flags and default dtype are preserved; default float64 callers still receive
  float32 models through train and reload.
- Official test and mask files are deliberately corrupted after manifest creation:
  normal-only fit still succeeds and its image reader opens train/good IDs only.
- CLI smoke trains a synthetic tree, saves ignored/outside-repo artifacts and
  reloads the checkpoint; existing output directories and invalid overrides fail.
- Corrupt/incompatible/incomplete/escaping/symlink/special checkpoint paths fail.
  Restricted CPU loading is asserted. Final-history failure retains an incomplete
  best checkpoint, cleans staging files and cannot publish completed weights.
- Independent id-reviewer verdict: **approved** after two reproduced issues
  (inherited torch dtype and premature final publication) were fixed and covered.
- Frontend/API routes and model catalog are unchanged; frontend checks were not
  repeated. Backend CI is configured for the CPU extra; remote CI was not run.
- One changed file per commit. The pre-existing untracked testing rule is excluded.
- Pending acceptance and next action:
  [`2026-10-14-wave-3-autoencoder.md`](../handoffs/2026-10-14-wave-3-autoencoder.md).
