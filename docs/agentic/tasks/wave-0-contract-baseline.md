# Task: `wave-0-contract-baseline`

- **Status:** done
- **Wave:** 0 — Foundation and agentic toolchain
- **Owner / role:** id-implementer
- **Model or category IDs:** `bottle`, `leather`, `metal_nut`, `autoencoder`, `patchcore`
- **Depends on:** ADR-001 through ADR-007
- **Allowed files:** repository scaffold, catalogs, API shell, dashboard shell,
  docs listed in the implementation plan, this card, and the workboard.

## Goal

Create the monorepo skeleton used by every later wave: catalogs, a FastAPI
shell, a dark dashboard shell, and CI. Do not train a model or download MVTec.

## Contract and conventions

Image and transport rules are ADR-004 and ADR-005. Category and model ids come
from `ml/configs/`. Inference routes validate input and return `501` while
models stay `planned`.

## Plan and acceptance criteria

- [x] Catalog YAML for categories, models, training defaults, and experiment protocols.
- [x] FastAPI health, dataset, model, and experiment routes.
- [x] Inference routes that validate uploads and refuse unready models.
- [x] Next.js shell with Overview, Datasets, Inference Lab, Experiments, and Comparison.
- [x] Ruff, Pytest, ESLint, Vitest, Playwright smoke, and GitHub Actions.
- [x] Architecture and API docs match the running contract.

## Tests and commands

```bash
python -m venv .venv
.venv/bin/python -m pip install -e "./backend[dev]"
.venv/bin/ruff check backend
.venv/bin/pytest backend
cd frontend && npm ci && npm run lint && npm run test && npm run build
```

## Non-goals

No PyTorch training, no Anomalib install, no MVTec download, no heatmap renderer,
no metric implementation, no checkpoint loading.

## Evidence / handoff

Local verification on 2026-10-14:

- `.venv/bin/ruff check backend` — passed.
- `.venv/bin/pytest backend` — 11 passed.
- `frontend` `npm run lint`, `npm run test` (3 passed), and `npm run build` — passed.
- Playwright smoke is wired in CI. It was not executed in this session because the browser binary was not installed locally.

No handoff. Wave 1 can start from `wave-1-dataset.md`.
