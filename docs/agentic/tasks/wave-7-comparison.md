# Task: `wave-7-comparison`

- **Status:** done (engineering); independently reviewed; real-data/CUDA gates open
- **Wave:** 7 — Comparison dashboard
- **Owner / role:** Codex (planner, implementer, test engineer); independent id-reviewer
- **Model or category IDs:** `autoencoder`, `patchcore`
- **Depends on:** Waves 5 and 6
- **Allowed files:** `frontend/features/{experiments,comparison}/`, comparison page,
  `frontend/lib/api.ts`, frontend tests and Playwright configuration,
  `backend/tests/inference_fixture.py` (test report export only), and Wave 7 docs.

## Goal

Compare two ready models on one upload and show saved experiment metrics side
by side, including latency and category filters.

## Contract and conventions

`POST /api/inference/compare` rejects identical model ids. Charts name the metric
and the split. The page reads experiment JSON produced by Wave 5.

## Plan and acceptance criteria

ADR-014: bounded local JSON selection; reports stay in browser memory. No new
endpoint, dependency or numerical computation. Compare within one saved run.

- [x] One image, two maps, two scores.
- [x] Metric table for the selected category.
- [x] Export of the visible table as CSV from the client.
- [x] Catalog-derived choices, distinct installed pairs and stale/error cleanup.
- [x] Reject malformed/incomplete reports and inconsistent test IDs; expose
      synthetic/partial scope, undefined reasons, timing scope and provenance.
- [x] Unit/browser evidence, lint/build and independent review.

## Tests and commands

```bash
cd frontend && npm run lint && npm run test && npm run build && npm run test:e2e
```

## Non-goals

No new training, metric computation, cross-run ranking, report server, dependency,
PDF report, experiment B/C or real-MVTec/CUDA acceptance claim.

## Evidence / handoff

- Frontend lint and 31 unit tests pass (27 report cases). Unit contract fixtures
  test two categories and selected-category export; they are not benchmark values.
- Production build passes, including TypeScript, with
  `NEXT_PUBLIC_API_URL=http://127.0.0.1:8001`.
- Seven browser cases pass with `PLAYWRIGHT_API_URL=http://127.0.0.1:8001
  PLAYWRIGHT_WEB_URL=http://localhost:3001 PLAYWRIGHT_PRODUCTION=1 npm run test:e2e`
  (7.6 s final run). Real-model synthetic comparison, delayed-call cleanup,
  actual evaluator JSON import/export and delayed-file-read cleanup are covered.
- `.venv/bin/pytest backend/tests/test_inference_runtime.py -q --tb=short`:
  38 passed, 45 existing upstream warnings (22.22 s). No full Python suite rerun
  claimed for this frontend wave. Ruff check and fixture format check pass.
- Independent `id-reviewer`: approved, no remaining findings. The review found
  impossible confusion counts were accepted; count/label membership checks and
  two regressions fixed it. Reviewer independently reran 31 unit tests and lint.
- Screenshots inspected; dataset/checkpoints/screenshots remain outside Git.
  User services 8000/3000 preserved; test servers 8001/3001 stopped.
- One file per commit. See
  `docs/agentic/handoffs/2026-10-14-wave-7-comparison.md` for the next boundary.
