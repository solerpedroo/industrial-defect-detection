"""Offline experiment A: freeze all normal thresholds, then compare sealed test IDs.

No training, HTTP, test-selected threshold, score scaling or hidden pixel sampling.
Generated reports contain IDs, hashes and numbers, never image/map payloads.
"""

import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from importlib.metadata import version
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
from PIL import Image

from ml.datasets import load_split_manifest
from ml.datasets.reader import _open_regular_file, file_checksum, inspect_category

from .calibration import calibrate_normal
from .config import EvaluationConfig, load_evaluation_config
from .metrics import evaluate_predictions

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json(path: Path, value: dict) -> None:
    _write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read(root: Path, image_id: str, *, mask: bool = False):
    path = root / image_id
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Source path contains a symlink.")
    with _open_regular_file(path) as stream:
        raw = stream.read()
    with Image.open(BytesIO(raw)) as image:
        image.load()
        if mask:
            array = np.asarray(image)
            if array.ndim != 2 or not np.isin(array, [0, 255]).all():
                raise ValueError("Expected a binary single-channel ground-truth mask.")
            array = np.asarray(array, dtype=np.uint8).copy()
        else:
            array = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return array, hashlib.sha256(raw).hexdigest()


def _load_artifact(path: Path, model_id: str, device: str):
    # Both public loaders enforce locally trusted regular .pt paths and state schemas.
    import torch

    from ml.models.autoencoder import load_autoencoder
    from ml.models.patchcore import load_patchcore

    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Artifact path contains a symlink.")
    digest = file_checksum(path)
    loader = load_autoencoder if model_id == "autoencoder" else load_patchcore
    predictor = loader(path, trusted_root=path.parent, device=device)
    with _open_regular_file(path) as stream:
        metadata = torch.load(stream, map_location="cpu", weights_only=True)["metadata"]
    if file_checksum(path) != digest:
        raise ValueError("Artifact changed while loading provenance.")
    metadata = dict(metadata)
    metadata["prepare_sidecar_sha256"] = None
    if model_id == "autoencoder":
        history_path = path.parent / "history.json"
        if history_path.exists() or history_path.is_symlink():
            with _open_regular_file(history_path) as stream:
                raw = stream.read()
            history = json.loads(raw)
            if (
                not isinstance(history, dict)
                or history.get("complete") is not True
                or history.get("checkpoint_sha256") != digest
                or any(history.get(key) != metadata[key] for key in ("config_hash", "split"))
            ):
                raise ValueError("Autoencoder history is not bound to this completed checkpoint.")
            metadata["duration_seconds"] = history.get("duration_seconds")
            metadata["prepare_sidecar_sha256"] = hashlib.sha256(raw).hexdigest()
            metadata["prepare_time_source"] = "checkpoint_bound_history_json"
        else:
            metadata["duration_seconds"] = None
            metadata["prepare_time_source"] = "unknown_missing_history_json"
    else:
        metadata["prepare_time_source"] = "checkpoint_metadata"
    duration = metadata.get("duration_seconds")
    if duration is not None and (
        type(duration) not in (int, float) or not np.isfinite(duration) or duration < 0
    ):
        raise ValueError("Invalid recorded preparation duration.")
    if duration is None and metadata["prepare_time_source"] != "unknown_missing_history_json":
        raise ValueError("Recorded preparation duration is missing.")
    return predictor, metadata, digest


def _preprocess(predictor):
    return predictor.preprocess if hasattr(predictor, "preprocess") else predictor.config.preprocess


def _predict(predictor, rgb, size):
    result = predictor.predict(rgb)
    anomaly_map = np.asarray(result.anomaly_map)
    score = float(result.score)
    if (
        anomaly_map.shape != (size, size)
        or anomaly_map.dtype != np.float32
        or not np.isfinite(anomaly_map).all()
        or (anomaly_map < 0).any()
        or not np.isfinite(score)
        or score < 0
    ):
        raise ValueError("Predictor violated finite nonnegative score/HW float32 map contract.")
    return score, anomaly_map.copy()


def _synchronize(device: str):
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def _memory(device: str) -> dict:
    if device.startswith("cuda"):
        import torch

        return {
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_memory_source": "cuda_allocator_peak_including_loaded_model",
        }
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        peak = None
    return {
        "peak_memory_bytes": peak,
        "peak_memory_source": "approximate_process_lifetime_high_water_rss_not_model_isolated",
    }


def run_experiment(
    dataset_root: Path,
    artifacts: dict[str, dict[str, Path]],
    output_dir: Path,
    *,
    data_kind: str,
    manifest_dir: Path | None = None,
    config: EvaluationConfig | None = None,
    device: str = "cpu",
) -> Path:
    """Compare trusted prepared artifacts on matching test IDs at model resolution.

    artifacts maps category -> model ID -> locally trusted checkpoint path.
    data_kind explicitly declares synthetic or mvtec_ad; it is not inferred from
    filenames. No population-FPR, original-resolution or industrial safety claim.
    """
    import torch

    from ml.models.autoencoder.prediction import resolve_device

    config = load_evaluation_config() if config is None else config
    if not isinstance(config, EvaluationConfig) or data_kind not in {"synthetic", "mvtec_ad"}:
        raise ValueError("Expected an evaluation config and explicit synthetic/mvtec_ad data kind.")
    device = str(resolve_device(device))
    root, output = Path(dataset_root), Path(output_dir).resolve()
    if set(artifacts) != set(config.categories) or any(
        set(paths) != set(config.models) for paths in artifacts.values()
    ):
        raise ValueError("Exactly both model artifacts are required for every requested category.")
    if output.is_relative_to(root.resolve()) or (
        output.is_relative_to(REPO_ROOT)
        and not any(
            output.is_relative_to(REPO_ROOT / name) and output != REPO_ROOT / name
            for name in ("runs", "artifacts", "checkpoints")
        )
    ):
        raise ValueError("Reports require an ignored/external run outside the source dataset.")
    output.mkdir(parents=True, exist_ok=False)
    manifests, frozen = {}, {}
    # This entire phase has no test inventory, image, mask or score reads.
    for category in config.categories:
        manifest = load_split_manifest(root, category, manifest_dir=manifest_dir)
        manifests[category] = manifest
        if len(manifest.validation_ids) * config.image_size**2 > config.max_pixels:
            raise ValueError("Validation pixels exceed the explicit metric budget.")
        frozen[category] = {}
        geometry = None
        for model_id in config.models:
            path = Path(artifacts[category][model_id]).absolute()
            predictor, metadata, digest = _load_artifact(path, model_id, device)
            pre = _preprocess(predictor)
            if (
                metadata["split"] != json.loads(json.dumps(manifest.to_dict()))
                or pre.category_id != category
                or pre.image_size != config.image_size
                or metadata["config"]["seed"] != config.seed
            ):
                raise ValueError(
                    "Artifact category, resolution, seed or normal split is incompatible."
                )
            current_geometry = (pre.image_size, pre.grayscale)
            if geometry is not None and geometry != current_geometry:
                raise ValueError(
                    "Comparison requires identical inference image geometry/color mode."
                )
            geometry = current_geometry
            if (
                data_kind == "mvtec_ad"
                and model_id == "patchcore"
                and not predictor.config.pre_trained
            ):
                raise ValueError("Real experiment A requires the pretrained PatchCore backbone.")
            scores, maps = [], []
            for image_id in manifest.validation_ids:
                rgb, checksum = _read(root, image_id)
                if checksum != manifest.train_checksums[image_id]:
                    raise ValueError("Validation bytes changed before calibration.")
                score, anomaly_map = _predict(predictor, rgb, config.image_size)
                scores.append(score)
                maps.append(anomaly_map)
            frozen[category][model_id] = {
                "image": calibrate_normal(scores, target_fpr=config.target_image_fpr).to_dict(),
                "pixel": calibrate_normal(
                    np.stack(maps), target_fpr=config.target_pixel_fpr
                ).to_dict(),
                "artifact_sha256": digest,
                "artifact_size_bytes": path.stat().st_size,
                "model_config_hash": metadata["config_hash"],
                "model_config": metadata["config"],
                "preprocessing_config_hash": pre.config_hash,
                "prepare_time_s": metadata["duration_seconds"],
                "prepare_time_source": metadata["prepare_time_source"],
                "prepare_sidecar_sha256": metadata["prepare_sidecar_sha256"],
                "pretrained_source": metadata.get("pretrained_source"),
                "preparation_environment": {
                    key: metadata.get(key) for key in ("device", "versions", "hardware")
                },
                "validation_ids": list(manifest.validation_ids),
                "validation_checksums": {
                    key: manifest.train_checksums[key] for key in manifest.validation_ids
                },
            }
            del predictor, maps
    calibration = {
        "schema_version": 1,
        "complete": False,
        "phase": "thresholds_frozen_before_test",
        "data_kind": data_kind,
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "calibration": frozen,
        "normal_splits": {key: value.to_dict() for key, value in manifests.items()},
    }
    _json(output / "calibration.json", calibration)
    calibration_digest = file_checksum(output / "calibration.json")
    rows, test_provenance = [], {}
    for category in config.categories:
        report = inspect_category(root, category)
        report.require_valid()
        samples = tuple(sample for sample in report.samples if sample.split == "test")
        if not samples or len(samples) * config.image_size**2 > config.max_pixels:
            raise ValueError("Empty test or test pixels exceed explicit metric budget.")
        masks, labels, source_hashes = [], [], {}
        for sample in samples:
            rgb, checksum = _read(root, sample.image_id)
            source_hashes[sample.image_id] = checksum
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            if sample.mask_id is not None:
                mask, checksum = _read(root, sample.mask_id, mask=True)
                source_hashes[sample.mask_id] = checksum
                if mask.shape != rgb.shape[:2]:
                    raise ValueError("Ground-truth mask and image geometry differ.")
            mask = (
                np.asarray(
                    Image.fromarray(mask).resize(
                        (config.image_size, config.image_size), Image.Resampling.NEAREST
                    )
                )
                > 0
            )
            masks.append(mask)
            labels.append(sample.is_anomalous)
        test_ids = [sample.image_id for sample in samples]
        test_provenance[category] = {"test_ids": test_ids, "source_checksums": source_hashes}
        for model_id in config.models:
            record = frozen[category][model_id]
            path = Path(artifacts[category][model_id]).absolute()
            if file_checksum(path) != record["artifact_sha256"]:
                raise ValueError("Artifact changed after validation thresholds were frozen.")
            predictor, current_metadata, digest = _load_artifact(path, model_id, device)
            if digest != record["artifact_sha256"]:
                raise ValueError("Artifact changed after validation thresholds were frozen.")
            if current_metadata["prepare_sidecar_sha256"] != record["prepare_sidecar_sha256"]:
                raise ValueError("Preparation history changed after thresholds were frozen.")
            warmup, checksum = _read(root, manifests[category].validation_ids[0])
            if checksum != record["validation_checksums"][manifests[category].validation_ids[0]]:
                raise ValueError("Warmup validation source changed.")
            for _ in range(config.warmup_images):
                _predict(predictor, warmup, config.image_size)
            _synchronize(device)
            if device.startswith("cuda"):
                import torch

                torch.cuda.reset_peak_memory_stats(device)
            scores, maps, timings = [], [], []
            for sample in samples:
                rgb, checksum = _read(root, sample.image_id)
                if checksum != source_hashes[sample.image_id]:
                    raise ValueError("Test image changed between compared model reads.")
                _synchronize(device)
                started = time.perf_counter()
                score, anomaly_map = _predict(predictor, rgb, config.image_size)
                _synchronize(device)
                timings.append((time.perf_counter() - started) * 1000)
                scores.append(score)
                maps.append(anomaly_map)
            memory = _memory(device)
            metrics = evaluate_predictions(
                labels,
                scores,
                np.stack(masks),
                np.stack(maps),
                image_threshold=record["image"]["threshold"],
                pixel_threshold=record["pixel"]["threshold"],
                fpr_limit=config.aupro_fpr_limit,
            )
            rows.append(
                {
                    "category": category,
                    "model_id": model_id,
                    "seed": config.seed,
                    "config_hash": config.config_hash,
                    "model_config_hash": record["model_config_hash"],
                    "test_ids": test_ids,
                    "labels": labels,
                    "scores": scores,
                    "metrics": metrics,
                    "thresholds": {key: record[key] for key in ("image", "pixel")},
                    "compute": {
                        **memory,
                        "prepare_time_s": record["prepare_time_s"],
                        "prepare_time_source": record["prepare_time_source"],
                        "artifact_size_bytes": record["artifact_size_bytes"],
                        "inference_latency_ms": float(np.mean(timings)),
                        "latency_median_ms": float(np.median(timings)),
                        "latency_p95_ms": float(np.quantile(timings, 0.95)),
                        "per_image_latency_ms": timings,
                        "throughput_images_per_s": 1000 / float(np.mean(timings)),
                        "timing_scope": "prediction_and_output_copy_no_file_decode_or_model_load",
                    },
                }
            )
            del predictor, maps
            if file_checksum(path) != record["artifact_sha256"]:
                raise ValueError("Artifact changed during test evaluation.")
            if (
                record["prepare_sidecar_sha256"] is not None
                and file_checksum(path.parent / "history.json") != record["prepare_sidecar_sha256"]
            ):
                raise ValueError("Preparation history changed during test evaluation.")
        final = inspect_category(root, category)
        final.require_valid()
        if final.samples != report.samples or any(
            file_checksum(root / key) != value for key, value in source_hashes.items()
        ):
            raise ValueError("Test inventory or source content changed during evaluation.")
    for category in config.categories:
        if load_split_manifest(root, category, manifest_dir=manifest_dir) != manifests[category]:
            raise ValueError("Normal source provenance changed during evaluation.")
    if file_checksum(output / "calibration.json") != calibration_digest:
        raise ValueError("Frozen calibration record changed during evaluation.")
    flat = []
    for row in rows:
        flat.append(
            {
                **{
                    key: row[key]
                    for key in ("category", "model_id", "seed", "config_hash", "model_config_hash")
                },
                **{
                    key: value
                    for key, value in row["metrics"].items()
                    if not isinstance(value, dict)
                },
                **{
                    key: value
                    for key, value in row["compute"].items()
                    if not isinstance(value, list)
                },
                "image_threshold": row["thresholds"]["image"]["threshold"],
                "pixel_threshold": row["thresholds"]["pixel"]["threshold"],
                "threshold_source": "normal_validation",
            }
        )
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
    _write(output / "results.csv", stream.getvalue())
    result = {
        **calibration,
        "complete": True,
        "phase": "official_test_report",
        "protocol_scope": "complete"
        if set(config.categories) == set(load_evaluation_config().categories)
        else "partial",
        "calibration_sha256": calibration_digest,
        "csv_sha256": file_checksum(output / "results.csv"),
        "tests": test_provenance,
        "results": rows,
        "environment": {
            "device": device,
            "device_name": torch.cuda.get_device_name(device)
            if device.startswith("cuda")
            else "CPU",
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                name: version(name)
                for name in ("numpy", "pillow", "torch", "torchvision", "anomalib", "timm")
            },
        },
    }
    _json(output / "results.json", result)
    return output / "results.json"
