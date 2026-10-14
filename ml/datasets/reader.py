"""Read MVTec metadata and audit local files without changing source data.

IDs are dataset-root-relative POSIX paths. RGB pixels are decoded only during
integrity inspection, never for threshold selection. No transforms occur here.
"""

import hashlib
import os
import re
import stat
import warnings
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DatasetIntegrityError(ValueError):
    """Source files cannot safely form a dataset partition."""


@dataclass(frozen=True)
class ImageSample:
    image_id: str
    split: Literal["train", "validation", "test"]
    defect_type: str
    mask_id: str | None = None

    @property
    def is_anomalous(self) -> bool:
        return self.defect_type != "good"


@dataclass(frozen=True)
class IntegrityIssue:
    code: str
    path: str
    detail: str


@dataclass(frozen=True)
class IntegrityReport:
    category_id: str
    samples: tuple[ImageSample, ...]
    issues: tuple[IntegrityIssue, ...]
    duplicates: tuple[tuple[str, ...], ...]
    train_checksums: dict[str, str]

    @property
    def valid(self) -> bool:
        return not self.issues and not self.duplicates

    def to_dict(self) -> dict:
        return {**asdict(self), "valid": self.valid}

    def require_valid(self) -> None:
        if not self.valid:
            raise DatasetIntegrityError(
                f"{self.category_id}: {len(self.issues)} integrity issue(s), "
                f"{len(self.duplicates)} duplicate group(s). Run the integrity audit."
            )


def category_path(dataset_root: Path, category_id: str) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", category_id):
        raise ValueError("Category IDs must be lowercase identifiers, not paths.")
    return Path(dataset_root) / category_id


def image_paths(directory: Path) -> tuple[Path, ...]:
    """List regular image files, excluding symlinks and linked directories."""
    if not directory.is_dir() or directory.is_symlink():
        return ()
    return tuple(
        sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and not any(parent.is_symlink() for parent in path.parents)
        )
    )


def _open_regular_file(path: Path) -> BinaryIO:
    """Open without following a final symlink or blocking on a substituted FIFO."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("Expected a regular image file.")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with _open_regular_file(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_category(dataset_root: Path, category_id: str) -> IntegrityReport:
    """Decode a category, verify masks, and find duplicate leakage.

    Duplicates within train or across official train/test fail. Repetition solely
    within test cannot influence validation selection. Masks are single-channel
    binary {0, 255} at the image resolution. Source files are never deleted.
    """
    root = Path(dataset_root)
    category = category_path(root, category_id)
    issues: list[IntegrityIssue] = []
    samples: list[ImageSample] = []
    masks: dict[tuple[str, str], str] = {}
    sizes: dict[str, tuple[int, int]] = {}
    pixels: dict[str, list[ImageSample]] = {}
    checksums: dict[str, str] = {}

    def issue(code: str, path: Path, detail: str) -> None:
        issues.append(IntegrityIssue(code, path.relative_to(root).as_posix(), detail))

    if not category.is_dir() or category.is_symlink():
        issue(
            "missing_category" if not category.exists() else "symlink",
            category,
            "Expected a regular category directory.",
        )
        return IntegrityReport(category_id, (), tuple(issues), (), {})

    for directory in (category / "train" / "good", category / "test" / "good"):
        if not directory.is_dir():
            issue("missing_directory", directory, "Required MVTec directory is missing.")
        elif not image_paths(directory):
            issue("empty_directory", directory, "Required image partition is empty.")

    for path in sorted(category.rglob("*")):
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            issue("symlink", path, "Dataset entries must not be symlinks.")
            continue
        parts = path.relative_to(category).parts
        if path.is_dir():
            allowed = (
                len(parts) == 1
                and parts[0] in {"train", "test", "ground_truth"}
                or len(parts) == 2
                and parts[0] in {"test", "ground_truth"}
                and (parts[0] != "ground_truth" or parts[1] != "good")
                or parts == ("train", "good")
            )
            if not allowed:
                issue("unexpected_directory", path, "Unexpected MVTec directory layout.")
            continue
        if not path.is_file():
            issue("nonregular_file", path, "Expected a regular image file.")
            continue
        if (
            len(parts) != 3
            or parts[0] not in {"train", "test", "ground_truth"}
            or parts[0] == "train"
            and parts[1] != "good"
            or parts[0] == "ground_truth"
            and parts[1] == "good"
            or path.suffix.lower() not in IMAGE_SUFFIXES
        ):
            issue("unexpected_file", path, "Expected a standard MVTec image path.")
            continue
        image_id = path.relative_to(root).as_posix()
        is_mask = parts[0] == "ground_truth"
        if is_mask:
            if not path.stem.endswith("_mask"):
                issue("mask_name", path, "Expected <image-stem>_mask.<extension>.")
                continue
            key = (parts[1], path.stem.removesuffix("_mask"))
            if key in masks:
                issue("ambiguous_mask", path, "Multiple masks match the same image stem.")
            masks[key] = image_id
        else:
            sample = ImageSample(image_id, parts[0], parts[1])
            samples.append(sample)
        try:
            # Hash exactly the bytes we decode, even if the source changes mid-audit.
            with _open_regular_file(path) as stream:
                raw_bytes = stream.read()
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(raw_bytes)) as image:
                    image.verify()
                with Image.open(BytesIO(raw_bytes)) as image:
                    image.load()
                    sizes[image_id] = image.size
                    if is_mask:
                        if len(image.getbands()) != 1 or not set(image.getdata()) <= {0, 255}:
                            issue("nonbinary_mask", path, "Expected a single-channel 0/255 mask.")
                    else:
                        digest = hashlib.sha256(
                            f"{image.width}x{image.height}:RGB:".encode()
                            + image.convert("RGB").tobytes()
                        ).hexdigest()
                        pixels.setdefault(digest, []).append(sample)
                        if sample.split == "train":
                            checksums[image_id] = hashlib.sha256(raw_bytes).hexdigest()
        except (
            OSError,
            ValueError,
            SyntaxError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ):
            issue("unreadable_image", path, "Image could not be safely verified and decoded.")

    paired: set[str] = set()
    with_masks: list[ImageSample] = []
    image_stems: set[tuple[str, str]] = set()
    for sample in samples:
        mask_id = None
        if sample.split == "test" and sample.is_anomalous:
            key = (sample.defect_type, Path(sample.image_id).stem)
            if key in image_stems:
                issue("ambiguous_image", root / sample.image_id, "Images share a mask stem.")
            image_stems.add(key)
            mask_id = masks.get(key)
            if mask_id is None:
                issue("missing_mask", root / sample.image_id, "Anomalous image has no mask.")
            else:
                paired.add(mask_id)
                if (
                    sample.image_id in sizes
                    and mask_id in sizes
                    and sizes[sample.image_id] != sizes[mask_id]
                ):
                    issue("mask_size", root / mask_id, "Mask and image dimensions differ.")
        with_masks.append(ImageSample(sample.image_id, sample.split, sample.defect_type, mask_id))
    for mask_id in sorted(set(masks.values()) - paired):
        issue("orphan_mask", root / mask_id, "Mask has no matching anomalous image.")
    duplicates = tuple(
        tuple(sample.image_id for sample in group)
        for group in pixels.values()
        if len(group) > 1 and any(sample.split == "train" for sample in group)
    )
    return IntegrityReport(category_id, tuple(with_masks), tuple(issues), duplicates, checksums)
