# Sample outputs

`scripts/render_report.py` writes protocol packages here, one directory per
protocol id: `report.md`, `results.json`, `results.csv` and an optional
`image_auroc.png`. Generated files are gitignored. The API reads this directory
unless `REPORTS_ROOT` points somewhere else. Commit only this note.
