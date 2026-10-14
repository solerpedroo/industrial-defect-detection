# Handoff: `wave-8-reports`

- **Date:** 2026-10-14
- **From / role:** id-implementer
- **To / role:** Wave 9 planner
- **Task card:** `docs/agentic/tasks/wave-8-reports.md`
- **State:** engineering complete; scientific acceptance remains partial

## Completed scope

Report packages are written offline as `report.md`, `results.json`,
`results.csv` and an optional `image_auroc.png`. `GET /api/reports/{id}`
returns the markdown and PNG names when the package exists under
`sample_outputs/` or `REPORTS_ROOT`. Unknown catalog ids, missing packages
and symlink packages return 404. Markdown larger than 1 MiB returns 413.
The handler does not train, score or calibrate.

Experiment B scores the same image ids under the recorded noise, blur and
brightness settings plus an original copy. Thresholds are copied from a
completed experiment A report whose source is `normal_validation`. Degraded
pixels stay in memory.

Experiment C records the seeded 25/50/75/100 percent draws and three repeats
over normal train ids. It does not retrain and does not report AUROC. Every
package repeats that scores, heatmap intensity and mask area are not physical
severity or financial loss.

## Exact evidence

- Commit/diff: uncommitted, relative to `a20a701`. New modules are
  `ml/experiments/{degrade,protocols,render,robustness,subsets}.py`,
  `scripts/render_report.py`, `backend/app/services/reports.py` and
  `backend/tests/test_reports.py`. Edited transport is
  `backend/app/{api/routes.py,core/config.py,schemas/catalog.py}`.
  Remaining edits record the wave (ADR-015, task/workboard, architecture,
  API contract, implementation plan, acceptance, playbook, experiment
  READMEs, `sample_outputs/README.md`, `AGENTS.md` and this handoff).
- `.venv/bin/ruff check backend ml/experiments scripts/render_report.py`: clean.
- `.venv/bin/pytest backend -q --tb=line`: 328 passed, 1 skipped, 45 warnings
  in 72.08s. The new file is `backend/tests/test_reports.py` (10 cases).
- `sample_outputs/*` stays gitignored except `sample_outputs/README.md`.
  No figure, checkpoint or MVTec image was committed.

## Decisions and conventions

ADR-015. Protocol B does not call calibration and does not write degraded
images into the dataset. Protocol C cannot persist a smaller fit manifest:
`load_split_manifest` still checksums every normal training file and rejects
a train set that is not the recorded complement of validation. The report
states that block. Catalog experiment statuses stay `planned`.

## Remaining risk or blocker

- Real MVTec preparation, experiment A on that copy, and CUDA remain open.
  Synthetic tests are engineering evidence only.
- No end-to-end robustness run loads a fitted Autoencoder or PatchCore
  artifact. `score_degraded_split` is a pure function; the CLI renders an
  existing JSON or a blocked document.
- Experiment C has a membership plan and no measured metrics. Closing that
  gate needs a split-loader change, which this wave does not make.
- Remote CI was not run.

## Next single action

Read and bound `docs/agentic/tasks/wave-9-ship.md`. Do not start it until this
diff is reviewed. Leave the real-data, CUDA and protocol C retrain gates open.
