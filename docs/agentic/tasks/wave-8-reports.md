# Task: `wave-8-reports`

- **Status:** done (engineering)
- **Wave:** 8 — Experiment reports
- **Owner / role:** id-implementer
- **Model or category IDs:** all
- **Depends on:** Wave 5, ADR-015
- **Allowed files:** `ml/experiments/`, `experiments/`, `scripts/render_report.py`,
  `GET /api/reports/{id}`, report tests, and the docs that record this wave.

## Goal

Render a Markdown report plus CSV, JSON, and PNG figures for protocols A, B, and C.
Protocol B applies the documented noise, blur, and brightness settings to the
same base images. Protocol C records training-subset membership. Retraining
those subsets stays blocked by the current manifest contract.

## Contract and conventions

Degradation parameters match `experiments/robustness/config.yaml`. Subset draws
use the recorded seed. Reports state limitations and refuse to call a score a
physical severity. Thresholds for B are copied from a completed experiment A
report. See ADR-015.

## Plan and acceptance criteria

- [x] `GET /api/reports/{id}` returns a generated report when the artifact exists.
- [x] Three protocols have a filled results section or an explicit blocked note.
- [x] Figures land in `sample_outputs/` and stay gitignored.

## Tests and commands

```bash
.venv/bin/ruff check backend ml/experiments scripts/render_report.py
.venv/bin/pytest backend
```

## Non-goals

No PDF renderer. No change to the split loader. No HTTP training. No claim that
experiment C measured AUROC.

## Evidence / handoff

- `.venv/bin/ruff check backend ml/experiments scripts/render_report.py`: clean.
- `.venv/bin/pytest backend -q --tb=line`: 328 passed, 1 skipped, 45 warnings in 72.08s.
- See `docs/agentic/handoffs/2026-10-14-wave-8-reports.md`.
