# Task: `wave-9-ship`

- **Status:** done (documentation)
- **Wave:** 9 — Hardening and ship
- **Owner / role:** id-implementer
- **Model or category IDs:** none
- **Depends on:** Waves 0–8, ADR-016
- **Allowed files:** `README.md`, `docs/mvp-acceptance.md`, `docs/deploy.md`,
  `docs/demo-storyboard.md`, CI, tests that close acceptance gaps, and the
  docs that record this wave.

## Goal

Make the repository reproducible for a reader who did not write it: English
README, acceptance checklist marked with evidence, demo storyboard, and a
deploy path that does not publish the dataset.

## Contract and conventions

Acceptance items flip only with a command or an artifact link. Real MVTec,
CUDA, and experiment C retraining stay open. See ADR-016.

## Plan and acceptance criteria

- [x] README quick start trains or loads the two models and opens the UI.
- [x] CI is green on the MVP tests. Local equivalent of `.github/workflows/ci.yml` passed. The GitHub Actions run was not queried in this session.
- [x] Acceptance table is updated.
- [x] No dataset images or checkpoints are tracked.

## Tests and commands

```bash
.venv/bin/ruff check backend/tests/test_ship_hygiene.py
.venv/bin/pytest backend -q --tb=line
cd frontend && npm run lint && npm run test && npm run build
cd frontend && CI=true npm run test:e2e
```

## Non-goals

PaDiM, EfficientAD, ONNX, video streams, and accounts. No live Vercel or
Railway deployment. No change to the default API image dependencies.

## Evidence / handoff

- `.venv/bin/ruff check backend/tests/test_ship_hygiene.py`: clean.
- `.venv/bin/pytest backend -q --tb=line`: 331 passed, 1 skipped, 45 warnings in 71.38s.
- `cd frontend && npm run lint && npm run test && npm run build`: lint clean, 31 Vitest tests passed, Next.js build succeeded.
- `cd frontend && CI=true npm run test:e2e`: 7 passed in 11.4s after installing Playwright Chromium. The first attempt failed because the browser binary was absent.
- See `docs/agentic/handoffs/2026-10-14-wave-9-ship.md`.
