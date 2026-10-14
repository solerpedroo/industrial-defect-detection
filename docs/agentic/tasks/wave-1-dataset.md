# Task: `wave-1-dataset`

- **Status:** proposed
- **Wave:** 1 — Dataset management
- **Owner / role:** unassigned
- **Model or category IDs:** `bottle`, `leather`, `metal_nut`
- **Depends on:** Wave 0
- **Allowed files:** `ml/datasets/`, `scripts/validate_dataset.py`, `backend/app/services/`,
  dataset tests, `docs/architecture.md` notes for the loader.

## Goal

Read a local MVTec AD tree, count images, reject unreadable files, and write a
validation manifest by holding out normal training images only.

## Contract and conventions

ADR-003 and ADR-007. Official `test/` stays sealed. Validation ids are stored
under `ml/data/` and gitignored.

## Plan and acceptance criteria

- [ ] Summaries distinguish train normal, validation normal, test normal, and test anomalous.
- [ ] Integrity check reports corrupt files without deleting them.
- [ ] Duplicate detection across splits fails the manifest build.
- [ ] API summary exposes the new counts.

## Tests and commands

```bash
.venv/bin/pytest backend/tests -q
```

Use synthetic directory trees. Do not commit MVTec images.

## Non-goals

No augmentation, no model, no pixel metrics.

## Evidence / handoff
