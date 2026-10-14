# Handoff: `wave-7-comparison`

- **Date:** 2026-10-14
- **From / role:** Codex, planner / implementer / test engineer
- **To / role:** Wave 8 planner
- **Task card:** `docs/agentic/tasks/wave-7-comparison.md`
- **State:** engineering complete; scientific acceptance remains partial

## Completed scope

Comparison now renders two installed calibrated models on the same upload, with
two maps/scores, shared overlay opacity, masks, timings and installed provenance.
Catalog-derived controls disallow identical/uninstalled pairs. Input changes,
errors, obsolete calls and unmount invalidate results.

Saved metrics read a local completed Wave 5 JSON up to 16 MiB, without uploading
or persisting it. The table compares both models within one run/category, and
CSV exports that category with recorded numbers, null reasons and provenance.

## Exact evidence

- Commit/diff: Wave 7 changes relative to `f2e5d68`, one file per commit. Source
  scope is `frontend/features/comparison/comparison-board.tsx`, the three
  `frontend/features/experiments/{report.ts,report.test.ts,saved-comparison.tsx}`
  files, comparison page, API cancellation signature, Playwright config/spec,
  and test-only `backend/tests/inference_fixture.py` report export. Remaining
  files document this same wave (ADR, task/workboard, README, architecture/API,
  implementation/acceptance/playbook, AGENTS and this handoff).
- `npm run lint`, `npm run test`: passed; 31 frontend tests (27 report cases).
- `NEXT_PUBLIC_API_URL=http://127.0.0.1:8001 npm run build`: passed, including TypeScript.
- `PLAYWRIGHT_API_URL=http://127.0.0.1:8001 PLAYWRIGHT_WEB_URL=http://localhost:3001
  PLAYWRIGHT_PRODUCTION=1 npm run test:e2e`: seven browser cases, real tiny
  synthetic models and actual generated evaluation report; no pretrained download.
  Includes held inference/file-read races and invalid/error replacement cleanup.
- `.venv/bin/pytest backend/tests/test_inference_runtime.py -q --tb=short`:
  38 passed, 45 existing upstream warnings. Numerical model code is unchanged;
  this wave does not claim a fresh run of the full 319-test Python suite.
- Ruff check over backend/ml/scripts and fixture format check passed; diff check
  passed. Browser screenshots inspected under ignored `frontend/test-results/`.
- Existing user services on ports 8000/3000 preserved; isolated checks use 8001/3001.
- Independent id-reviewer approved after fixing impossible confusion counts;
  reviewer independently reran 31 frontend tests and lint. Final browser run:
  seven passed in 7.6 s after the count-consistency fix and production rebuild.

## Decisions and conventions

ADR-014 distinguishes saved official-test metrics from live upload output.
Structural checks cover completion/schema, catalog IDs, full row coverage,
ordered test IDs and label agreement, source hash presence, finite/ranged numbers,
explicit null reasons, confusion-count/label membership and frozen threshold
consistency. Hashes are recorded identifiers, not an authenticated trust chain;
the browser does not rehash model artifacts or recalculate scientific metrics.

The UI makes no cross-run ranking or shared score/color-scale claim. CSV fields
are escaped and spreadsheet formula prefixes neutralized. Full precision is
preserved; nulls become empty value cells with explicit reasons, never zero.
Preparation/memory conventions and warmed prediction timing are displayed.

## Remaining risk or blocker

- Real MVTec preparation/evaluation and CUDA remain unverified; synthetic success
  is engineering evidence only. No licensed images/checkpoints were committed.
- Large reports above 16 MiB require a separately scoped pagination/serving design.
  Local reports may contain edited/misleading values despite structural checks.
- Wave 8's report endpoint remains 404; Markdown/PNG report packages and B/C runs
  have not been implemented. Remote CI was not run in this task.
- Upstream Python 3.14 TorchScript/deprecation warnings are retained.

## Next single action

Read and bound `docs/agentic/tasks/wave-8-reports.md` before implementing the
report/science wave; preserve real-data gates and the saved/live distinction.
