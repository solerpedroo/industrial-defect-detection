# Task: `wave-7-comparison`

- **Status:** proposed
- **Wave:** 7 — Comparison dashboard
- **Owner / role:** unassigned
- **Model or category IDs:** `autoencoder`, `patchcore`
- **Depends on:** Waves 5 and 6
- **Allowed files:** `frontend/features/experiments/`, comparison route, chart components.

## Goal

Compare two ready models on one upload and show saved experiment metrics side
by side, including latency and category filters.

## Contract and conventions

`POST /api/inference/compare` rejects identical model ids. Charts name the metric
and the split. The page reads experiment JSON produced by Wave 5.

## Plan and acceptance criteria

- [ ] One image, two maps, two scores.
- [ ] Metric table for the selected category.
- [ ] Export of the visible table as CSV from the client.

## Tests and commands

```bash
cd frontend && npm run test && npm run test:e2e
```

## Non-goals

No new training and no PDF report.

## Evidence / handoff
