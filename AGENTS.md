# Industrial Defect Detection — agent entrypoint

## Current boundary

**Waves 1–4 engineering is implemented; real-data acceptance remains open.**
The user authorized continuation with synthetic evidence (ADR-009/010/011).
Wave 5 is the next planned task. Before real-data training, audit the licensed
local MVTec copy and create normal-only manifests. Offline Autoencoder fitting
and checkpoint prediction are implemented with CPU synthetic evidence.
Frozen pretrained PatchCore memory and trusted offline reload are also implemented;
see `docs/agentic/handoffs/2026-10-14-wave-4-patchcore.md` for provenance,
family-specific raw score scales and pending acceptance context.
Calibrated metrics and HTTP model inference are not implemented yet; both catalog
models stay `planned`. Read [`docs/mvp-acceptance.md`](docs/mvp-acceptance.md)
before treating any module as done. Status lives in [`docs/agentic/workboard.md`](docs/agentic/workboard.md).

## Start protocol

1. Read `docs/project-scope.md`, then `docs/implementation-plan.md`.
2. Read `docs/agentic/architecture-decisions.md` and the assigned task card.
   If no card exists, create one from `task-template.md` before code.
3. Claim one bounded task only. Do not mix unrelated models or waves.
4. Read the applicable rule in `.cursor/rules/` and skill in `.cursor/skills/`.
5. Implement, validate, document a handoff, and stop at the task boundary.

## Shared invariants

- Training uses **normal images only**. Anomalous images and ground-truth masks
  are evaluation data. Official MVTec test images never define a threshold.
- The model catalog in `ml/configs/models.yaml` and the category catalog in
  `ml/configs/categories.yaml` are the source of IDs, labels, and starting
  hyperparameters. Do not duplicate those defaults in the UI.
- `ml/` owns numerical work. FastAPI routes adapt transport. React renders state.
  HTTP handlers never train a model.
- Anomaly scores are not physical severity and are not financial loss.
  Bounding boxes, when added, come from anomaly masks, not from a separate detector.
- UI and user-facing technical copy are English. Planning docs may be Portuguese
  or English. Do not translate API identifiers.
- MVTec AD is CC BY-NC-SA 4.0. Do not commit dataset images, masks, or
  redistributable copies.
- Do not add a dependency, endpoint, or data model outside the assigned task
  without recording an ADR and getting the Planner to split the work.

## Context that survives agents

Use repository artifacts, not chat memory, as the handoff protocol:

- `docs/agentic/architecture-decisions.md` — accepted cross-cutting choices.
- `docs/agentic/tasks/` — one committed, bounded task card per implementation task.
- `docs/agentic/handoffs/` — committed factual handoffs for incomplete work or
  non-obvious follow-up. Do not write a handoff for a clean, completed task.
- `docs/agentic/workboard.md` — current wave status and ownership; edit only
  your row, and avoid simultaneous broad rewrites.

Never treat an uncommitted file, terminal history, or a previous agent message
as authoritative context.

## Required handoff content

When work is incomplete or needs review, copy `handoff-template.md` into the
`handoffs/` directory as `YYYY-MM-DD-<task-id>.md`. Include: exact commit/diff
scope, decisions made, commands and results, remaining risks, and the next
single action. Never include secrets, base64 image payloads, or large logs.

## Roles

Use the specialized agents in `.cursor/agents/`:

- `id-planner` owns slicing, dependencies, and task cards.
- `id-implementer` owns one vertical slice (data, model, metric, or UI).
- `id-test-engineer` owns numerical test design and regression evidence.
- `id-reviewer` owns contract, safety, leakage, and scientific-review gates.

The same person or agent may fill several roles sequentially; do not ask two
agents to edit the same source files concurrently.

## Completion gate

Before a task is ready for review: run the commands specified by its task card,
update its DoD, update its workboard row, and create a factual handoff if the
next person needs context. A reviewer may reject scope creep even if tests pass.
