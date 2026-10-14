# Handoff: `wave-1-dataset`

- **Date:** 2026-10-14
- **From / role:** Codex / implementer and test engineer; independent id-reviewer approved
- **To / role:** id-planner / local dataset owner
- **Task card:** `docs/agentic/tasks/wave-1-dataset.md`
- **State:** partial — implementation verified, licensed real-data acceptance pending

## Completed scope

Local MVTec reader, full integrity audit, normal-only deterministic validation
manifests, validated partition consumers, CLI, existing summary API extension,
typed dataset view, Python/CLI CI lint, regression tests and documentation.
No model, preprocessing, threshold, new endpoint or new dependency.

## Exact evidence

- Commit/diff: the Wave 1 commit containing this handoff. Exact source scope:
  `ml/datasets/{__init__,reader,splits,summary}.py`, `scripts/validate_dataset.py`,
  `backend/app/{schemas,services}/catalog.py`, `backend/tests/test_dataset.py`,
  `frontend/types/api.ts`, `frontend/features/datasets/dataset-browser.tsx`,
  `frontend/e2e/shell.spec.ts`, `.github/workflows/ci.yml`. Documentation scope:
  `AGENTS.md`, `README.md`, `data/README.md`, `ml/data/README.md`,
  `docs/{architecture,api-contract,implementation-plan,mvp-acceptance,agent-playbook}.md`,
  `docs/agentic/{architecture-decisions,workboard}.md`, the Wave 1 card and this handoff.
- `.venv/bin/pytest backend/tests -q`: 72 passed.
- `.venv/bin/ruff check --config backend/pyproject.toml backend ml scripts`: clean.
- Frontend lint/build: pass; Vitest: 3 passed.
- Browser flow against isolated API 8011 / frontend 3011: 1 passed, checking
  actual validation fields and rendered counts. Exact commands are in the card.
- Default CLI audit: exit 1, all three real MVTec categories missing; no real
  split files created. No licensed images or generated artifacts enter the commit.

## Decisions and conventions

ADR-008 records version-1 normal-only manifests at `ml/data/splits/<category>.json`.
Seeded SHA-256 ranking picks `ceil(n * fraction)` normal validation IDs; both
partitions must be nonempty. Manifests record raw training checksums. Reads reject
changed content, unexpected training entries and malformed/special manifest files.
An audit reads test pixels/masks only for corruption, mask pairing and duplicate
checks. Normal partition consumption and summaries never decode test pixels.
Source files are unchanged. Writes are atomic per category with explicit overwrite.

## Remaining risk or blocker

The three-category licensed MVTec tree is absent. Synthetic tests establish code
behavior but do not satisfy real-data preparation acceptance. Keep the Wave 1
local-data gate and MVP checklist open until the real audit/split creation pass.
No real model accuracy or runtime claim is supported.

The API hashes training files on summary reads to avoid stale validity decisions;
large real datasets may warrant measured caching work in a separately planned task.
Dataset content must stay immutable during an experiment; a manifest check is
not a permanent snapshot. Existing API processes need restarting to load the new
schema; user-owned running processes were not interrupted.

## Next single action

Place a licensed local MVTec AD copy at `data/mvtec-ad/` and run
`.venv/bin/python scripts/validate_dataset.py --create-split --validation-fraction 0.2 --json`.
The fraction is an explicit experiment choice, not a tuned default. Record the
successful audit and counts for all three categories, close the local-data gate,
then claim the existing Wave 2 card.
