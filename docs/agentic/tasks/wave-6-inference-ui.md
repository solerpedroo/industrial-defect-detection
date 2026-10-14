# Task: `wave-6-inference-ui`

- **Status:** proposed
- **Wave:** 6 — Inference lab
- **Owner / role:** unassigned
- **Model or category IDs:** models marked `ready`
- **Depends on:** Wave 5
- **Allowed files:** `backend/app/inference/`, `backend/app/visualization/`,
  `frontend/features/inference/`, inference API tests, e2e.

## Goal

Let a user upload one image, pick a category and a ready model, and see the
score, binary decision, heatmap, overlay with opacity, and a mask download.

## Contract and conventions

ADR-005 and ADR-006. Bounding boxes are optional contours of the mask.
`inference_ms` excludes decode and PNG encode.

## Plan and acceptance criteria

- [ ] `501` remains for models that are still `planned`.
- [ ] Ready models return the JSON contract in `docs/api-contract.md`.
- [ ] UI states API errors and does not keep a stale map after a failed request.

## Tests and commands

```bash
.venv/bin/pytest backend
cd frontend && npm run test && npm run test:e2e
```

## Non-goals

No multi-experiment dashboard (Wave 7).

## Evidence / handoff
