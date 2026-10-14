"""Startup-only trusted frozen-model registry; uploads never fit or calibrate.

Predictors consume HWC RGB uint8 and recorded model-family normalization. Maps
are square HW float32 nonnegative raw scores, not probabilities or severity.
Model/config/calibration bytes are snapshots; restart to install a new version.
"""

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Lock

import numpy as np

from ml.datasets.reader import IMAGE_SUFFIXES, _open_regular_file, file_checksum
from ml.datasets.splits import SplitManifest, _select
from ml.evaluation import EvaluationConfig


def _safe_file(path: Path, *, limit: int) -> bytes:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Serving paths must not contain symlinks.")
    with _open_regular_file(path) as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Serving JSON exceeds the configured size envelope.")
    return raw


def _relative(root: Path, value: str) -> Path:
    if not isinstance(value, str):
        raise ValueError("Serving paths must be relative POSIX strings.")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or str(pure) != value
        or not pure.parts
        or "\\" in value
    ):
        raise ValueError("Serving path escapes or is not canonical.")
    path = root.joinpath(*pure.parts)
    if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError("Serving path escapes the trusted root.")
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Serving paths must not contain symlinks.")
    return path


def _split(document: dict, category: str) -> None:
    if (
        not isinstance(document, dict)
        or set(document) != set(SplitManifest.__dataclass_fields__)
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["category_id"] != category
    ):
        raise ValueError("Invalid normal split provenance.")
    checksums = document["train_checksums"]
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("Normal source checksums are required.")
    for image_id, digest in checksums.items():
        path = PurePosixPath(image_id)
        if (
            len(path.parts) != 4
            or path.parts[:3] != (category, "train", "good")
            or str(path) != image_id
            or path.suffix.lower() not in IMAGE_SUFFIXES
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("Expected normal-only source IDs and SHA-256 checksums.")
    train, validation = _select(
        sorted(checksums), document["seed"], document["validation_fraction"]
    )
    if document["train_ids"] != list(train) or document["validation_ids"] != list(validation):
        raise ValueError("Normal partition membership does not match its recorded split.")


def _threshold(document: dict, count: int, target: float) -> float:
    fields = {
        "threshold",
        "target_fpr",
        "quantile",
        "sample_count",
        "observed_fpr",
        "source",
        "method",
        "comparator",
    }
    if not isinstance(document, dict) or set(document) != fields:
        raise ValueError("Invalid frozen threshold fields.")
    for key in ("threshold", "target_fpr", "quantile", "observed_fpr"):
        value = document[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Frozen threshold numbers must be finite.")
    if (
        not 0 <= document["threshold"] <= np.finfo(np.float32).max
        or type(document["sample_count"]) is not int
        or document["sample_count"] != count
        or document["target_fpr"] != target
        or document["quantile"] != 1 - target
        or not 0 <= document["observed_fpr"] <= target
        or document["source"] != "normal_validation"
        or document["method"] != "higher"
        or document["comparator"] != ">"
    ):
        raise ValueError("Threshold is not the recorded normal-validation operating point.")
    return float(document["threshold"])


@dataclass(frozen=True)
class InstalledModel:
    predictor: object
    image_threshold: float
    pixel_threshold: float
    data_kind: str
    artifact_sha256: str
    calibration_sha256: str
    config_hash: str
    size: int


class ServingRegistry:
    def __init__(self):
        self._models: dict[tuple[str, str], InstalledModel] = {}
        self._lock = Lock()

    def categories(self, model_id: str, *, synthetic: bool = False) -> list[str]:
        return sorted(
            category
            for (category, model), installed in self._models.items()
            if model == model_id and (not synthetic or installed.data_kind == "synthetic")
        )

    def require(self, category: str, model: str) -> InstalledModel:
        try:
            return self._models[category, model]
        except KeyError as error:
            raise LookupError(
                "No calibrated artifact is installed for this category/model."
            ) from error

    def predict(self, category: str, model: str, rgb: np.ndarray):
        installed = self.require(category, model)
        with self._lock:
            device = installed.predictor.device
            if device.type == "cuda":
                import torch

                torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = installed.predictor.predict(rgb)
            if device.type == "cuda":
                import torch

                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - started) * 1000
        return prediction, installed, elapsed


def load_registry(manifest_path: Path | None, *, allow_synthetic: bool = False) -> ServingRegistry:
    registry = ServingRegistry()
    if manifest_path is None:
        return registry
    try:
        manifest = json.loads(_safe_file(Path(manifest_path).absolute(), limit=1024 * 1024))
        if (
            set(manifest) != {"schema_version", "trusted_root", "device", "entries"}
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
        ):
            raise ValueError("Unsupported serving manifest schema.")
        root = Path(manifest["trusted_root"])
        if (
            not root.is_absolute()
            or not root.is_dir()
            or root == Path(root.anchor)
            or root.is_symlink()
            or any(parent.is_symlink() for parent in root.parents)
        ):
            raise ValueError("An explicit regular trusted artifact root is required.")
        if not isinstance(manifest["entries"], list) or not manifest["entries"]:
            raise ValueError("Configured serving needs at least one entry.")
        # These optional imports occur only when the operator installs a manifest.
        import torch

        from ml.models.autoencoder.prediction import resolve_device

        device = str(resolve_device(manifest["device"]))
        for entry in manifest["entries"]:
            if not isinstance(entry, dict) or set(entry) != {
                "category_id",
                "model_id",
                "checkpoint",
                "calibration",
            }:
                raise ValueError("Invalid serving entry fields.")
            category, model = entry["category_id"], entry["model_id"]
            if (category, model) in registry._models:
                raise ValueError("Duplicate serving category/model.")
            checkpoint = _relative(root, entry["checkpoint"])
            calibration_path = _relative(root, entry["calibration"])
            raw = _safe_file(calibration_path, limit=16 * 1024 * 1024)
            document = json.loads(raw)
            if (
                type(document["schema_version"]) is not int
                or document["schema_version"] != 1
                or document["complete"] is not False
                or document["phase"] != "thresholds_frozen_before_test"
            ):
                raise ValueError(
                    "Expected Wave 5 frozen normal calibration, not test-selected values."
                )
            config = EvaluationConfig(**document["config"])
            if (
                category not in config.categories
                or model not in config.models
                or document["config_hash"] != config.config_hash
            ):
                raise ValueError(
                    "Serving category/model or calibration config hash is incompatible."
                )
            kind = document["data_kind"]
            if kind not in {"synthetic", "mvtec_ad"} or kind == "synthetic" and not allow_synthetic:
                raise ValueError("Synthetic serving requires explicit demonstration opt-in.")
            digest = file_checksum(checkpoint)
            if model == "autoencoder":
                from ml.models.autoencoder import load_autoencoder

                loader = load_autoencoder
            else:
                from ml.models.patchcore import load_patchcore

                loader = load_patchcore
            predictor = loader(checkpoint, trusted_root=root, device=device)
            with _open_regular_file(checkpoint) as stream:
                metadata = torch.load(stream, map_location="cpu", weights_only=True)["metadata"]
            pre = predictor.preprocess if model == "autoencoder" else predictor.config.preprocess
            split = metadata["split"]
            _split(split, category)
            record = document["calibration"][category][model]
            validation = split["validation_ids"]
            if (
                record["artifact_sha256"] != digest
                or record["model_config_hash"] != metadata["config_hash"]
                or record["model_config"] != metadata["config"]
                or document["normal_splits"][category] != split
                or record["validation_ids"] != validation
                or record["validation_checksums"]
                != {key: split["train_checksums"][key] for key in validation}
                or pre.category_id != category
                or pre.image_size != config.image_size
                or metadata["config"]["seed"] != config.seed
                or record["preprocessing_config_hash"] != pre.config_hash
            ):
                raise ValueError(
                    "Artifact, preprocessing or held-out provenance does not match calibration."
                )
            if model == "patchcore" and kind == "mvtec_ad" and not predictor.config.pre_trained:
                raise ValueError("MVTec serving requires pretrained PatchCore.")
            image_threshold = _threshold(record["image"], len(validation), config.target_image_fpr)
            pixel_threshold = _threshold(
                record["pixel"], len(validation) * pre.image_size**2, config.target_pixel_fpr
            )
            # Warm with generated RGB only; no dataset, fitting or calibration reads.
            predictor.predict(np.zeros((pre.image_size, pre.image_size, 3), dtype=np.uint8))
            if (
                file_checksum(checkpoint) != digest
                or file_checksum(calibration_path) != hashlib.sha256(raw).hexdigest()
            ):
                raise ValueError("Serving artifacts changed during startup.")
            registry._models[category, model] = InstalledModel(
                predictor,
                image_threshold,
                pixel_threshold,
                kind,
                digest,
                hashlib.sha256(raw).hexdigest(),
                metadata["config_hash"],
                pre.image_size,
            )
        return registry
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as error:
        raise ValueError(
            "Invalid serving installation; check manifest and model/calibration compatibility."
        ) from error
