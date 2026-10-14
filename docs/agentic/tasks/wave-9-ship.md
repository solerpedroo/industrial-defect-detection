# Task: `wave-9-ship`

- **Status:** proposed
- **Wave:** 9 — Hardening and ship
- **Owner / role:** unassigned
- **Model or category IDs:** none
- **Depends on:** Waves 0–8
- **Allowed files:** `README.md`, `docs/mvp-acceptance.md`, `docs/deploy.md`,
  CI, tests that close acceptance gaps.

## Goal

Make the repository reproducible for a reader who did not write it: English
README, acceptance checklist marked with evidence, demo storyboard, and a
deploy path that does not publish the dataset.

## Contract and conventions

Acceptance items in `docs/mvp-acceptance.md` flip to done only with a command
or artifact link. Video capture follows `docs/demo-storyboard.md`.

## Plan and acceptance criteria

- [ ] README quick start trains or loads the two models and opens the UI.
- [ ] CI is green on the MVP tests.
- [ ] Acceptance table is updated.
- [ ] No dataset images or checkpoints are tracked.

## Tests and commands

```bash
.venv/bin/ruff check backend && .venv/bin/pytest backend
cd frontend && npm run lint && npm run test && npm run build && npm run test:e2e
```

## Non-goals

PaDiM, EfficientAD, ONNX, video streams, and accounts.

## Evidence / handoff
