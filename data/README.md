# Local datasets

Place a local MVTec AD copy at `data/mvtec-ad/`. This directory is gitignored
except for this file.

Expected layout for each selected category:

```text
data/mvtec-ad/bottle/train/good/
data/mvtec-ad/bottle/test/good/
data/mvtec-ad/bottle/test/<defect-type>/
data/mvtec-ad/bottle/ground_truth/<defect-type>/
```

The same pattern applies to `leather` and `metal_nut`.

MVTec AD is licensed CC BY-NC-SA 4.0 for non-commercial use. Do not commit or
redistribute the images. Download instructions are on the
[official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).

Validate a local copy with:

```bash
.venv/bin/python scripts/validate_dataset.py
```

The audit exits with status 1 for missing categories, unreadable images, invalid
mask associations, duplicate train/test content or stale manifests. It reports
problems without changing source files. Tests use tiny synthetic images only.

After a successful audit, explicitly hold out normal training images:

```bash
.venv/bin/python scripts/validate_dataset.py --create-split --validation-fraction 0.2
```

The fraction is an explicit experiment choice. The seed defaults to the canonical
`ml/configs/training_defaults.yaml` value. Manifests live at
`ml/data/splits/<category>.json`, outside Git. Add `--overwrite` only when you
intend to replace an existing split. Keep the data unchanged during experiments.

For an external copy or a single-category inspection:

```bash
.venv/bin/python scripts/validate_dataset.py --dataset-root /path/to/mvtec-ad --category bottle --json
```

External-root manifests are for offline use; the web API reads the repository
copy at `data/mvtec-ad/`. No dataset download or redistribution is automated.
