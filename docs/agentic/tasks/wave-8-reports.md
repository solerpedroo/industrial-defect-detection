# Task: `wave-8-reports`

- **Status:** proposed
- **Wave:** 8 — Experiment reports
- **Owner / role:** unassigned
- **Model or category IDs:** all
- **Depends on:** Wave 5
- **Allowed files:** `ml/experiments/`, `experiments/`, `scripts/render_report.py`,
  `GET /api/reports/{id}`.

## Goal

Render a Markdown report plus CSV, JSON, and PNG figures for protocols A, B, and C.
Protocol B applies the documented noise, blur, and brightness settings to the
same base images. Protocol C repeats training on 25, 50, 75, and 100 percent
normal subsets.

## Contract and conventions

Degradation parameters match `experiments/robustness/config.yaml`. Subset draws
use the recorded seed. Reports state limitations and refuse to call a score a
physical severity.

## Plan and acceptance criteria

- [ ] `GET /api/reports/{id}` returns a generated report when the artifact exists.
- [ ] Three protocols have a filled results section or an explicit blocked note.
- [ ] Figures land in `sample_outputs/` and stay gitignored.

## Tests and commands

```bash
.venv/bin/pytest backend
```

## Non-goals

No PDF renderer.

## Evidence / handoff
