# Handoff: `wave-2-preprocessing`

- **Date:** 2026-10-14
- **From / role:** Codex / implementer and test engineer; independent id-reviewer approved
- **To / role:** id-planner / Wave 3 implementer
- **Task card:** `docs/agentic/tasks/wave-2-preprocessing.md`
- **State:** partial — engineering complete; real-data and model-consumer acceptance pending

## Completed scope

Canonical YAML-backed preprocessing config, stage-aware RGB resize/normalization,
optional grayscale, explicitly permitted and seeded training horizontal flip,
synchronized binary mask geometry, JSON-safe provenance and before/after helpers.
No model, HTTP route, UI, new dependency or metric. The user authorized advancing
Wave 2 with synthetic evidence and requested one changed file per commit.

## Exact evidence

- Commit/diff scope: one file per Wave 2 commit following repository baseline `a6c982f`.
  Source: `ml/preprocessing/{config,pipeline,__init__}.py`,
  `ml/configs/{categories,training_defaults}.yaml`, `backend/tests/test_preprocessing.py`.
  Docs: `AGENTS.md`, `README.md`, `docs/{architecture,implementation-plan,mvp-acceptance,agent-playbook}.md`,
  `docs/agentic/{architecture-decisions,workboard}.md`, the Wave 2 card and this handoff.
- `.venv/bin/pytest backend/tests -q`: 121 passed (72 previous + 49 preprocessing).
- `.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts`: clean.
- `git diff --check`: clean. No frontend code changed, so frontend checks were
  not repeated for this numerical-only slice.
- Independent reviewer: approved, no concrete findings. JSON record replay also
  reproduced augmented grayscale output, paired masks and metadata exactly.

## Decisions and conventions

ADR-009 defines RGB HWC uint8 [0, 255] input and contiguous `(3, size, size)`
float32 normalized output. Current defaults are ImageNet RGB statistics from
YAML. Grayscale replicates Pillow L to RGB before channel normalization. Images
use bilinear resize; 0/255 masks use nearest-neighbor. Both stretch to a square.

`apply_preprocess` is shared by every stage. Default category permissions deny
flips. Train flipping needs both explicit category permission and positive
probability, plus an explicit seed. Non-train stages never augment or accept
training seeds. No global RNG is modified; no dataset files are selected or read.
Load product settings with `load_preprocess_config`, preserving the category's
reviewed permission when deriving explicit experiment configurations.

Each result carries display RGB, optional paired mask, model_input, and metadata
with config hash, seed, operation decisions and library versions. Persist its
`record` beside future experiment/checkpoint artifacts, not dataset images in Git.

## Remaining risk or blocker

Real MVTec preparation remains open. Actual training and model inference callers
do not yet exist; stage tests are not evidence of a trained model. Wave 3 must
choose and record autoencoder normalization compatible with its decoder output;
ImageNet defaults are not an automatic prescription for reconstruction loss.

Seed derivation belongs to the future training loop. Reusing one seed for every
sample repeats its flip decision; use reproducible per-sample/per-epoch seeds.
Previews are numerical display helpers, not an implemented HTTP/UI feature.

## Next single action

Claim the existing Wave 3 task with the Planner and specify its effective
normalization and seed derivation. Implement model-boundary conversion from
`PreprocessResult.model_input` to PyTorch using this shared function, and retain
the recorded config with artifacts. Audit and split the licensed local MVTec copy
before any real-data fit; keep test data out of model selection.
