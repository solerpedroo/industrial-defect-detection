# Task: `wave-1-dataset`

- **Status:** implemented and reviewed — real-data acceptance pending
- **Wave:** 1 — Dataset management
- **Owner / role:** Codex — planner, implementer, test engineer; independent id-reviewer
- **Model or category IDs:** `bottle`, `leather`, `metal_nut`
- **Depends on:** Wave 0
- **Allowed files:** `ml/datasets/`, `scripts/validate_dataset.py`, `backend/app/services/`,
  `backend/app/schemas/catalog.py`, dataset tests, `frontend/types/api.ts`,
  `frontend/features/datasets/dataset-browser.tsx`, `.github/workflows/ci.yml`,
  `frontend/e2e/shell.spec.ts` (dataset API-to-view check),
  dataset/API/architecture documentation and wave status records.

Planner clarification: the existing summary endpoint needs an additive count and
split status; its schema and typed consumer belong to this same vertical slice.
CI must lint the ML and CLI code as well as the API. No new endpoint or dependency.

## Goal

Read a local MVTec AD tree, count images, reject unreadable files, and write a
validation manifest by holding out normal training images only.

## Contract and conventions

ADR-003 and ADR-007. Official `test/` stays sealed. Validation ids are stored
under `ml/data/` and gitignored.

ADR-008 defines the versioned manifest, deterministic normal-only membership,
content provenance, fail-closed handling of stale manifests, and additive summary
fields. Integrity inspection may read test pixels/masks solely to find corruption,
mask association errors, or duplicate content, never for model selection.

## Plan and acceptance criteria

- [x] Summaries distinguish train normal, validation normal, test normal, and test anomalous.
- [x] Integrity check reports corrupt files without deleting them.
- [x] Duplicate detection across splits fails the manifest build.
- [x] API summary exposes the new counts.
- [x] Manifest writes are atomic, deterministic, and refuse accidental replacement.
- [x] Changed training content or malformed manifests cannot report a valid split.
- [x] CLI gets category IDs and default seed from the canonical YAML files.
- [x] Dataset UI renders validation count and split status from the API.

## Tests and commands

```bash
.venv/bin/pytest backend/tests -q
.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts
cd frontend && npm run lint && npm run test && npm run build
cd frontend && npm run test:e2e
```

Use synthetic directory trees. Do not commit MVTec images.

## Non-goals

No augmentation, no model, no pixel metrics.
No dataset download. Real-data acceptance remains open until a licensed local
copy of all three categories is validated.

## Evidence / handoff

- `.venv/bin/pytest backend/tests -q`: **72 passed** (11 baseline + 61 dataset cases).
- `.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts`: clean.
- Frontend `npm run lint`: clean; `npm run test`: **3 passed**; `npm run build`: success.
- Browser evidence: built with `NEXT_PUBLIC_API_URL=http://127.0.0.1:8011`,
  served with `npm run start -- --port 3011`, and ran
  `PLAYWRIGHT_API_URL=http://127.0.0.1:8011 PLAYWRIGHT_WEB_URL=http://localhost:3011 npm run test:e2e`:
  **1 passed**. The API on 8011 used
  `ALLOWED_ORIGINS=http://localhost:3011,http://127.0.0.1:3011`.
  The test checks the actual dataset response contains integer validation counts
  before checking rendering. Existing servers on 8000/3000 were not stopped;
  they can carry old loaded API code. Verification servers were stopped afterward
  and the default frontend build restored.
- Synthetic fixtures: 4×3 RGB/L uint8 images in [0, 255], normal holdout seed 42;
  seed-42 bottle validation IDs are `001.png` and `002.png` at fraction 0.4.
  All three synthetic categories also pass the CLI with ceil-rounded fraction 0.25.
- Leakage evidence: disjoint fit/validation/test IDs; test pixel changes do not
  alter membership; normal partition consumers and summaries do not decode images.
- Regression evidence: corrupt images, missing/orphan/wrong-size/nonbinary
  masks, copies with different PNG encoding, stale/tampered/extreme manifests,
  source replacement during audit, special files, and concurrent manifest writers.
- Independent `id-reviewer` verdict: **approved** after reproduced source-safety
  and malformed-manifest issues were fixed and covered by regression tests.
  No training in HTTP, test tuning, dependency, endpoint, or model changes.
- `.venv/bin/python scripts/validate_dataset.py --json`: **exit 1**, correctly
  reporting missing Bottle, Leather and Metal Nut; no real-data manifest generated.
- Real-data gate: **open**. Engineering criteria above have synthetic evidence,
  not measurements or acceptance on licensed MVTec images. Next action is in
  [`2026-10-14-wave-1-dataset.md`](../handoffs/2026-10-14-wave-1-dataset.md).
