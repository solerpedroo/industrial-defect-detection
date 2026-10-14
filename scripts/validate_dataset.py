"""Print local MVTec category counts. Does not download or copy the dataset."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.datasets import summarize_category  # noqa: E402

CATEGORIES = ("bottle", "leather", "metal_nut")
DATASET_ROOT = REPO_ROOT / "data" / "mvtec-ad"


def main() -> int:
    print(f"Dataset root: {DATASET_ROOT}")
    for category_id in CATEGORIES:
        summary = summarize_category(DATASET_ROOT, category_id)
        print(
            f"{category_id}: {summary['availability']} "
            f"train={summary['train_normal']} "
            f"test_normal={summary['test_normal']} "
            f"test_anomalous={summary['test_anomalous']} "
            f"masks={summary['mask_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
