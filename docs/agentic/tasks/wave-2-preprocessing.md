# Task: `wave-2-preprocessing`

- **Status:** done — engineering; real-data and model-consumer acceptance pending
- **Wave:** 2 — Image preprocessing
- **Owner / role:** Codex — planner, implementer, test engineer; independent id-reviewer
- **Model or category IDs:** all catalog categories
- **Depends on:** Wave 1 implementation; real-data gate remains open
- **Allowed files:** `ml/preprocessing/`, `ml/configs/training_defaults.yaml`,
  `ml/configs/categories.yaml` (explicit category augmentation permissions),
  `backend/tests/test_preprocessing.py`, ADR/task/workboard/handoff records,
  `AGENTS.md`, `README.md`, architecture/implementation/acceptance/playbook docs.

Planner decision: on 2026-10-14 the user explicitly authorized continuing Wave 2
without closing the real-MVTec gate. Validate the reusable code on synthetic
arrays and do not claim real-data or model integration acceptance. One commit per
changed file, including documentation; preserve the pre-existing untracked rule.

## Goal

Provide one recorded preprocess path for training and inference: resize to the
configured square size, RGB normalization, optional grayscale, and a log of
the operations applied.

## Contract and conventions

ADR-004. Augmentations are opt-in per category. Horizontal flips are disabled
unless the category config allows them. Geometric changes that would move a
defect mask must be applied to the mask with the same parameters.

ADR-009 defines a single stage-aware NumPy/Pillow pipeline, three-channel grayscale,
CHW float32 output, synchronized nearest-neighbor masks, recorded configuration
and operation provenance, and explicit seeds for optional training augmentation.
Only horizontal flip is implemented; all category permissions default to false.

## Plan and acceptance criteria

- [x] Config dataclass matches `training_defaults.yaml` image size.
- [x] Before/after preview helper exists for the future UI.
- [x] Train and inference call the same function (stage paths; model consumers deferred).
- [x] Invalid dtype, shape, normalization, category policy, stage, seed or mask fails explicitly.
- [x] Validation/test/inference never apply training augmentation.
- [x] Paired masks retain binary values and follow the same geometric decisions.
- [x] Records serialize to JSON and include config hash, operations and library versions.

The shared-function criterion means both stage paths are exercised through
`apply_preprocess`; actual model training/inference consumers arrive in Waves 3/4.

## Tests and commands

```bash
.venv/bin/pytest backend/tests -q
.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts
```

## Non-goals

No training loop and no automatic claim that every augmentation is valid.
No new dependency, endpoint, UI, model, robustness experiment or real dataset download.

## Evidence / handoff

- `.venv/bin/pytest backend/tests -q`: **121 passed** (72 existing + 49 preprocessing).
- `.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts`: **clean**.
- `git diff --check`: **clean**.
- Fixtures: tiny RGB/mask uint8 arrays in [0, 255]; train seeds 2 and 42,
  with paired flip/no-flip assertions. Global NumPy state is preserved.
- Hand-computed cases verify per-channel normalization, grayscale primary luma
  and bilinear center interpolation. Mask nearest-neighbor values stay 0/255.
- Default train/validation/test/inference numerics are identical for the same
  pixels. A PNG decoded through the API boundary produces the same normalized
  input as the normal train stage without augmentation. No production model exists.
- The independent id-reviewer approved the implementation: **no concrete findings**.
  It additionally replayed an augmented grayscale train record after JSON
  round-trip and obtained identical model arrays, paired masks and metadata.
- No new dependency, HTTP route, UI, training code, threshold or measured model score.
- Commits: exactly one changed file per commit for this task; the pre-existing
  untracked `.cursor/rules/testing-anomaly.mdc` is excluded.
- Real MVTec preparation and actual model consumers remain pending. Integration
  details are in [`2026-10-14-wave-2-preprocessing.md`](../handoffs/2026-10-14-wave-2-preprocessing.md).
