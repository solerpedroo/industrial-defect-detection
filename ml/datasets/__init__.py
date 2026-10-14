"""MVTec AD directory reader.

The reader counts files. It does not import images into Git and it does not
decide thresholds. Validation manifests are a Wave 1 deliverable.
"""

from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def count_images(directory: Path) -> int:
    """Count image files under ``directory``. Missing directories count as zero."""
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def summarize_category(dataset_root: Path, category_id: str) -> dict[str, object]:
    """Summarize one MVTec category directory.

    ``availability`` is ``present`` when the category folder exists. Counts use
    the official ``train/good``, ``test/good``, other ``test`` defect folders,
    and ``ground_truth`` masks. This function does not create a validation split.
    """
    category_root = dataset_root / category_id
    if not category_root.is_dir():
        return {
            "availability": "missing",
            "train_normal": 0,
            "test_normal": 0,
            "test_anomalous": 0,
            "mask_count": 0,
            "defect_types": [],
            "validation_split": "not_created",
        }

    defect_types: list[dict[str, object]] = []
    test_root = category_root / "test"
    if test_root.is_dir():
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            if defect_dir.name == "good":
                continue
            images = count_images(defect_dir)
            defect_types.append({"id": defect_dir.name, "test_images": images})

    anomalous = sum(int(item["test_images"]) for item in defect_types)
    return {
        "availability": "present",
        "train_normal": count_images(category_root / "train" / "good"),
        "test_normal": count_images(category_root / "test" / "good"),
        "test_anomalous": anomalous,
        "mask_count": count_images(category_root / "ground_truth"),
        "defect_types": defect_types,
        "validation_split": "not_created",
    }
