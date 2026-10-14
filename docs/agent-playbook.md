# Agent playbook

Operational companion to [`../AGENTS.md`](../AGENTS.md). Use [`agentic/`](agentic/) records instead of a previous conversation.

## Loop for every bounded task

1. **Plan:** the Planner creates a card in `agentic/tasks/`, checks dependencies, and declares allowed files.
2. **Implement:** one Implementer claims it, follows the applicable rules and skills, and makes the smallest vertical change.
3. **Test:** the Test Engineer records exact commands, fixtures, seeds, and tolerances in the card.
4. **Review:** the Reviewer checks the plan, ADRs, diff, and evidence. Leakage of the test split or an HTTP training path blocks the card.
5. **Handoff:** leave a factual handoff only for partial, blocked, or non-obvious work. Update the owner's workboard row.

## Parallelism protocol

Parallelize independent model packages and tests only after the Planner assigns disjoint files. Serialize edits to shared points: catalogs, Pydantic schemas, the inference adapter, the typed API client, and the app shell. No two agents edit the same card or workboard row.

## Per-wave gate

- Task-card DoD items are checked.
- Local CI commands pass with recorded output.
- ADRs cover any new cross-cutting convention.
- The Reviewer confirms the wave does not start later features.
- Continuation can start from the committed card alone.

## Wave progress

| Wave | Status | Task card | Last verified |
| --- | --- | --- | --- |
| 0 | done | `wave-0-contract-baseline.md` | 2026-10-14 |
| 1 | implemented; real-data gate open | `wave-1-dataset.md` | 2026-10-14 |
| 2 | done (engineering; consumers pending) | `wave-2-preprocessing.md` | 2026-10-14 |
| 3 | done (engineering; real-data/API pending) | `wave-3-autoencoder.md` | 2026-10-14 |
| 4 | proposed | `wave-4-patchcore.md` | — |
| 5 | proposed | `wave-5-evaluation.md` | — |
| 6 | proposed | `wave-6-inference-ui.md` | — |
| 7 | proposed | `wave-7-comparison.md` | — |
| 8 | proposed | `wave-8-reports.md` | — |
| 9 | proposed | `wave-9-ship.md` | — |

When a wave is done, update this table, the workboard row, and record command output in the task card.
