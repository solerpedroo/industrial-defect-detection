"""Read-only file counts and validated normal partition membership."""

from pathlib import Path

from .reader import category_path, image_paths
from .splits import ManifestError, load_split_manifest, manifest_path


def summarize_category(
    dataset_root: Path, category_id: str, *, manifest_dir: Path | None = None
) -> dict[str, object]:
    """Report stale manifests explicitly without repairing them.

    Counts describe inventory, not an integrity certificate. Test pixels and
    masks are not decoded here. Valid manifests exclude validation IDs from fit.
    """
    category = category_path(Path(dataset_root), category_id)
    present = category.is_dir() and not category.is_symlink()
    train_normal = len(image_paths(category / "train" / "good")) if present else 0
    validation_normal = 0
    validation_split = "not_created"
    path = manifest_path(category_id, manifest_dir)
    if path.exists() or path.is_symlink():
        try:
            manifest = load_split_manifest(
                Path(dataset_root), category_id, manifest_dir=manifest_dir
            )
            train_normal = len(manifest.train_ids)
            validation_normal = len(manifest.validation_ids)
            validation_split = "created"
        except (ManifestError, FileNotFoundError):
            validation_split = "invalid"
    test = category / "test"
    defect_types = (
        [
            {"id": directory.name, "test_images": len(image_paths(directory))}
            for directory in sorted(test.iterdir())
            if directory.is_dir() and not directory.is_symlink() and directory.name != "good"
        ]
        if present and test.is_dir() and not test.is_symlink()
        else []
    )
    return {
        "availability": "present" if present else "missing",
        "train_normal": train_normal,
        "validation_normal": validation_normal,
        "test_normal": len(image_paths(test / "good")) if present else 0,
        "test_anomalous": sum(item["test_images"] for item in defect_types),
        "mask_count": len(image_paths(category / "ground_truth")) if present else 0,
        "defect_types": defect_types,
        "validation_split": validation_split,
    }
