"""Versioned, deterministic normal-only partitions; no test-based selection."""

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .reader import (
    IMAGE_SUFFIXES,
    DatasetIntegrityError,
    ImageSample,
    _open_regular_file,
    category_path,
    file_checksum,
    image_paths,
    inspect_category,
)

DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"
SCHEMA_VERSION = 1


class ManifestError(ValueError):
    """A manifest is malformed or no longer matches normal training files."""


@dataclass(frozen=True)
class SplitManifest:
    schema_version: int
    category_id: str
    seed: int
    validation_fraction: float
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    train_checksums: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def manifest_path(category_id: str, manifest_dir: Path | None = None) -> Path:
    category_path(Path("."), category_id)
    directory = DEFAULT_MANIFEST_DIR if manifest_dir is None else Path(manifest_dir)
    return directory / f"{category_id}.json"


def _select(ids: list[str], seed: int, fraction: float) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ManifestError("Seed must be an integer in [0, 2**32).")
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ManifestError("Validation fraction must be finite and strictly between 0 and 1.")
    count = math.ceil(len(ids) * fraction)
    if not 0 < count < len(ids):
        raise ManifestError("Split requires at least one fit image and one validation image.")
    ranked = sorted(
        ids,
        key=lambda image_id: (hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest(), image_id),
    )
    held_out = set(ranked[:count])
    return tuple(sorted(set(ids) - held_out)), tuple(sorted(held_out))


def create_split_manifest(
    dataset_root: Path,
    category_id: str,
    *,
    seed: int,
    validation_fraction: float,
    manifest_dir: Path | None = None,
    overwrite: bool = False,
) -> SplitManifest:
    """Audit first, then atomically persist a normal-only validation holdout.

    Sources are never moved. Existing manifests require explicit overwrite.
    No test IDs, masks or absolute paths enter the manifest.
    """
    destination = manifest_path(category_id, manifest_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists for '{category_id}'; use --overwrite.")
    report = inspect_category(Path(dataset_root), category_id)
    report.require_valid()
    train_ids, validation_ids = _select(sorted(report.train_checksums), seed, validation_fraction)
    manifest = SplitManifest(
        SCHEMA_VERSION,
        category_id,
        seed,
        validation_fraction,
        train_ids,
        validation_ids,
        report.train_checksums,
    )
    for image_id, checksum in manifest.train_checksums.items():
        if file_checksum(Path(dataset_root) / image_id) != checksum:
            raise DatasetIntegrityError("Normal training content changed during the audit.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(manifest.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Same-directory link publishes atomically and cannot clobber a concurrent writer.
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def load_split_manifest(
    dataset_root: Path, category_id: str, *, manifest_dir: Path | None = None
) -> SplitManifest:
    """Validate schema, membership and raw checksums, opening only train files.

    Raises FileNotFoundError when absent and ManifestError when invalid. Fitting
    must never fall back to held-out images on load failure.
    """
    path = manifest_path(category_id, manifest_dir)
    if path.is_symlink():
        raise ManifestError("Split manifest must not be a symlink.")
    try:
        with _open_regular_file(path) as stream:
            raw = stream.read().decode("utf-8")
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError) as error:
        raise ManifestError("Cannot read split manifest.") from error
    try:
        document = json.loads(raw)
        if not isinstance(document, dict) or set(document) != set(
            SplitManifest.__dataclass_fields__
        ):
            raise ManifestError("Unexpected split manifest fields.")
        if (
            type(document["schema_version"]) is not int
            or document["schema_version"] != SCHEMA_VERSION
        ):
            raise ManifestError("Unsupported manifest version.")
        if document["category_id"] != category_id:
            raise ManifestError("Manifest category does not match the request.")
        checksums = document["train_checksums"]
        if not isinstance(checksums, dict) or not all(
            isinstance(key, str)
            and isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for key, value in checksums.items()
        ):
            raise ManifestError("Invalid training checksums.")
        for field in ("train_ids", "validation_ids"):
            if not isinstance(document[field], list) or not all(
                isinstance(item, str) for item in document[field]
            ):
                raise ManifestError("Partition IDs must be string lists.")
        category = category_path(Path(dataset_root), category_id)
        train_root = category / "train" / "good"
        if category.is_symlink() or (category / "train").is_symlink() or train_root.is_symlink():
            raise ManifestError("Training directories must not be symlinks.")
        for entry in (category / "train").rglob("*"):
            if (
                entry.is_symlink()
                or entry.is_dir()
                and entry != train_root
                or not entry.is_dir()
                and (
                    not entry.is_file()
                    or entry.parent != train_root
                    or entry.suffix.lower() not in IMAGE_SUFFIXES
                )
            ):
                raise ManifestError("Unexpected or symlinked training entry added after the audit.")
        paths = image_paths(train_root)
        ids = [image.relative_to(dataset_root).as_posix() for image in paths]
        if any(image.parent != train_root for image in paths) or set(ids) != set(checksums):
            raise ManifestError("Normal training inventory changed or contains nested images.")
        train_ids, validation_ids = _select(ids, document["seed"], document["validation_fraction"])
        if (
            tuple(document["train_ids"]) != train_ids
            or tuple(document["validation_ids"]) != validation_ids
        ):
            raise ManifestError("Partition membership does not match the recorded protocol.")
        if any(
            file_checksum(image) != checksums[image.relative_to(dataset_root).as_posix()]
            for image in paths
        ):
            raise ManifestError("Normal training content changed since manifest creation.")
        return SplitManifest(
            SCHEMA_VERSION,
            category_id,
            document["seed"],
            document["validation_fraction"],
            train_ids,
            validation_ids,
            checksums,
        )
    except (ValueError, TypeError, KeyError, OSError, RecursionError, OverflowError) as error:
        if isinstance(error, ManifestError):
            raise
        raise ManifestError("Malformed or unreadable split manifest.") from error


def partition_samples(
    dataset_root: Path,
    category_id: str,
    split: Literal["train", "validation", "test"],
    *,
    manifest_dir: Path | None = None,
) -> tuple[ImageSample, ...]:
    """Return fit/validation membership or audited official test image-mask pairs.

    Fit and validation require a valid manifest and never open test pixels.
    Samples carry IDs, not transformed tensors; preprocessing is Wave 2.
    """
    if split == "test":
        report = inspect_category(Path(dataset_root), category_id)
        report.require_valid()
        return tuple(sample for sample in report.samples if sample.split == "test")
    if split not in {"train", "validation"}:
        raise ValueError("Split must be train, validation or test.")
    manifest = load_split_manifest(Path(dataset_root), category_id, manifest_dir=manifest_dir)
    ids = manifest.train_ids if split == "train" else manifest.validation_ids
    return tuple(ImageSample(image_id, split, "good") for image_id in ids)
