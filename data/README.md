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
