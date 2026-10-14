# Task: `wave-5-evaluation`

- **Status:** done (engineering; real-MVTec experiment A/CUDA/API gates open)
- **Wave:** 5 — Evaluation and benchmark
- **Owner / role:** Codex — planner, implementer, test engineer; independent id-reviewer
- **Model or category IDs:** `autoencoder`, `patchcore`
- **Depends on:** Waves 3 and 4
- **Allowed files:** `ml/evaluation/`, `experiments/autoencoder-vs-patchcore/`,
  `scripts/evaluate_experiment.py`, `backend/tests/test_evaluation*.py`, ADR,
  task/workboard/handoff and README/architecture/acceptance/plan/playbook/AGENTS docs.

Planner clarification (baseline `5e8b820`): implement one offline evaluation
slice, with no new dependency. NumPy owns exact tied-score metrics and normal
quantile calibration. A bounded experiment A runner loads both trusted artifact
families, freezes all validation thresholds before auditing/reading any test,
then evaluates identical test IDs and nearest-neighbor resized masks at the same
model resolution. JSON/CSV go to a new ignored/external run. Synthetic evidence
may complete engineering; three-category real-MVTec numbers remain open while
licensed data is absent. Preserve one changed file per commit. No HTTP/UI work.

## Goal

Compute image-level and pixel-level metrics on the official test split after
thresholds are frozen on validation. Run experiment A on Bottle, Leather, and
Metal Nut.

## Contract and conventions

ADR-003 and ADR-006. Record latency with warmup. Record approximate peak memory
and artifact size. Export CSV and JSON under the experiment directory, gitignored
if they embed images.

## Plan and acceptance criteria

- [x] Image AUROC, AUPR, precision, recall, F1, confusion counts.
- [x] Pixel AUROC, AUPR, IoU, Dice, and implemented AUPRO.
- [x] Threshold source is written next to the metrics.
- [x] Same test ids for both models.
- [x] All category/model normal quantiles freeze before any test audit/read.
- [x] Exact tied-score cases, undefined metrics and source/artifact drift fail safely.
- [x] Warmed timing, honest peak-memory scope and preparation/artifact provenance.
- [x] New ignored/external JSON/CSV runs; completed JSON publishes last.
- [ ] Real licensed Bottle/Leather/Metal Nut experiment A report (data absent).

## Tests and commands

```bash
.venv/bin/pytest backend/tests -q
PATCHCORE_PRETRAINED_SMOKE=1 .venv/bin/pytest backend/tests -q --tb=short
.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts
.venv/bin/python -m pip check
```

Synthetic score arrays must match hand-computed AUROC cases.

## Non-goals

No degradation study (experiment B) and no subset study (experiment C).

## Evidence / handoff

- Full suite with actual pretrained Wave 4 smoke: **281 passed**, 45 retained
  upstream warnings, 46.10s. Evaluation targeted suite: **68 passed**, 17.10s.
- Default full suite: **280 passed, 1 skipped**, 45 warnings, 45.33s; only
  the opt-in pretrained smoke is skipped.
- Independent id-reviewer repeated the 68 targeted tests, lint and whitespace
  checks; final verdict approved with no remaining code findings.
- Three review findings were corrected and covered: Autoencoder duration is in
  its checksum-bound history, not checkpoint; lossy float64 score conversion
  cannot create ties; full category coverage is independent of category order.
- Forty-seven numerical/config cases include hand AUROC .75/AP 5/6, tied/reversed/
  single-class scores, strict higher quantiles, empirical exceedance, undefined
  denominator reasons, equal-region AUPRO/clipped endpoints/8-connectivity and
  50 seed-42 randomized tied-score comparisons to an independent sklearn oracle.
- Twenty-one integration cases use six 32×32 synthetic normals per category,
  seed 42, validation fraction .33 (four fit/two held-out), tiny real one-epoch
  Autoencoder and real Anomalib with explicit random backbone. Tests verify the
  first audit sees all thresholds, identical test IDs, masks/test values cannot
  affect calibration, artifact compatibility, history/source drift, budgets,
  unknown timing provenance, failed publication and CLI behavior.
- Full three-category engineering evidence is **synthetic**, not a real experiment
  A report. No benchmark result image, checkpoint or dataset was committed.
- API plus `ml.evaluation` imports load neither torch nor Anomalib in a fresh
  interpreter; optional libraries are loaded only by the experiment runner.
- Ruff, diff whitespace and pip dependency checks pass. No frontend/HTTP code,
  model readiness, dependency or training implementation changed.
- Environment remains Python 3.14.8, torch 2.10.0+cpu, torchvision 0.25.0+cpu,
  Anomalib 2.7.0, timm 1.0.30, NumPy 2.5.4, Pillow 11.3.0. Upstream TorchScript/
  deprecated-import warnings remain visible; CUDA/remote CI unverified.
- One changed file per commit; continuation context:
  `docs/agentic/handoffs/2026-10-14-wave-5-evaluation.md`.
