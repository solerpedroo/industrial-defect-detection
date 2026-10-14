# Task: `wave-4-patchcore`

- **Status:** done (engineering; real-data/CUDA/API gates open)
- **Wave:** 4 — PatchCore integration
- **Owner / role:** Codex — planner, implementer, test engineer; independent id-reviewer
- **Model or category IDs:** `patchcore`
- **Depends on:** Wave 1, ADR that adds Anomalib
- **Allowed files:** `ml/models/patchcore/`, `ml/models/__init__.py` (common predictor contract),
  `scripts/build_patchcore.py`, `backend/tests/test_patchcore.py`,
  `ml/configs/training_defaults.yaml`, `backend/pyproject.toml`, `.github/workflows/ci.yml`,
  ADR/task/workboard/handoff and architecture/README/implementation/acceptance/playbook/AGENTS docs.

Planner clarification: reuse Anomalib's low-level PatchcoreModel to avoid its
Lightning calibration/evaluation pipeline opening official test data. Add an
optional `patchcore` extra and CPU torchvision CI setup. HTTP routes/adapters
stay deferred. User authorized the next wave with existing real-data gates open;
one file per commit continues to apply. Source baseline: `28f6208`.

## Goal

Wrap Anomalib PatchCore in a project adapter. Build a normal-patch memory with
ResNet-18 and return an anomaly score plus a spatial map. Expose the same
prediction dataclass as the autoencoder.

## Contract and conventions

The backbone is not fine-tuned while the memory is built. Coreset sampling
starts from the ratio in `training_defaults.yaml`.

ADR-011 specifies ImageNet-normalized RGB via the shared pipeline, frozen/eval
ResNet-18 layer2/layer3 features, seeded k-center coreset, and uncalibrated raw
distance score/maps. Both algorithms return the common AnomalyPrediction contract.

## Plan and acceptance criteria

- [x] Adapter does not import FastAPI.
- [x] Memory artifact path is outside Git.
- [x] A tiny synthetic run distinguishes a constant image from a corrupted one.
- [x] Shared interface documents score and map shapes.
- [x] Normal fit IDs only populate memory; validation/test pixels and masks are never decoded.
- [x] Saved backbone state and memory reload without downloading weights again.
- [x] Configuration, versions, seed, source checksums and memory size are recorded.
- [x] Invalid/small coreset, source drift, nonfinite state and unsafe artifacts fail explicitly.
- [x] No API-ready status, threshold, HTTP route or new model family enters scope.

## Tests and commands

```bash
.venv/bin/pytest backend/tests -q
.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts
PATCHCORE_PRETRAINED_SMOKE=1 .venv/bin/pytest backend/tests -q
.venv/bin/python -m pip check
```

## Non-goals

No PaDiM, no EfficientAD, no full MVTec benchmark (Wave 5).
No automatic dataset download, HTTP inference or score calibration. Real MVTec
and CUDA evidence remain separate from synthetic CPU engineering acceptance.

## Evidence / handoff

- Full suite with pretrained smoke: **213 passed**, 45 upstream warnings, 36.66s.
- Default suite: **212 passed, 1 skipped**, 45 warnings, 35.82s; only the
  explicitly enabled pretrained smoke is skipped.
- Independent id-reviewer repeated all **43 PatchCore tests** with pretrained
  smoke; approved without concrete findings. Ruff and whitespace checks pass.
- API import in a fresh interpreter loads neither torch nor Anomalib; pip check
  reports no broken requirements. No frontend/HTTP implementation was changed.
- Fixtures: six synthetic 32×32 normal RGB images, seed 42, normal validation
  fraction 0.33; four fit images give 64 patches and a 32×384 memory at ratio 0.5.
  Tests separately exercise explicit random mode and actual pretrained weights.
- Evidence includes frozen parameters/BatchNorm, exact fit bytes/no held-out
  decoding, same-environment determinism and restored RNG/default dtype,
  offline reload, weighted score semantics, budget/config guards, corrupt state,
  path/FIFO rejection, CLI and failed final publication.
- Verified environment: Python 3.14.8, torch 2.10.0+cpu, torchvision 0.25.0+cpu,
  Anomalib 2.7.0, timm 1.0.30, NumPy 2.5.4, Pillow 11.3.0.
- Known warnings originate upstream: TorchScript on Python 3.14+, timm layer
  imports and Anomalib dinov2 imports. They are not suppressed globally.
- One changed file per commit; artifacts/dataset/cache remain outside Git.
- Non-obvious continuation context:
  `docs/agentic/handoffs/2026-10-14-wave-4-patchcore.md`.
