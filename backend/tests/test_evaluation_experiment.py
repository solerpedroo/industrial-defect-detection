"""Synthetic CPU seed-42 integration of both real offline model consumers."""

import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from ml.datasets import create_split_manifest
from ml.evaluation import evaluate, experiment, load_evaluation_config
from ml.models.patchcore import build_patchcore_memory, load_patchcore_config
from ml.training.autoencoder import train_autoencoder
from ml.training.config import load_autoencoder_config


@pytest.fixture(autouse=True)
def cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _rgb(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (value, value + 1, value + 2)).save(path)


@pytest.fixture
def prepare(tmp_path):
    def make(categories=("bottle",)):
        root, manifests = tmp_path / "dataset", tmp_path / "splits"
        artifacts, splits = {}, {}
        for category in categories:
            for index in range(6):
                _rgb(root / category / "train/good" / f"{index:03}.png", 40 + index)
            _rgb(root / category / "test/good/000.png", 100)
            _rgb(root / category / "test/broken/000.png", 180)
            mask = root / category / "ground_truth/broken/000_mask.png"
            mask.parent.mkdir(parents=True)
            Image.new("L", (32, 32), 255).save(mask)
            splits[category] = create_split_manifest(
                root, category, seed=42, validation_fraction=0.33, manifest_dir=manifests
            )
            ae = load_autoencoder_config(category)
            ae = replace(
                ae,
                preprocess=replace(ae.preprocess, image_size=32),
                channels=(2, 4, 8),
                max_epochs=1,
                batch_size=2,
            )
            ae_run = train_autoencoder(
                root, category, tmp_path / f"{category}-ae", manifest_dir=manifests, config=ae
            )
            pc = load_patchcore_config(category)
            pc = replace(
                pc,
                preprocess=replace(pc.preprocess, image_size=32),
                pre_trained=False,
                batch_size=2,
                coreset_sampling_ratio=0.5,
            )
            pc_run = build_patchcore_memory(
                root, category, tmp_path / f"{category}-pc", manifest_dir=manifests, config=pc
            )
            artifacts[category] = {
                "autoencoder": ae_run.checkpoint_path,
                "patchcore": pc_run.checkpoint_path,
            }
        return SimpleNamespace(
            root=root,
            manifests=manifests,
            splits=splits,
            artifacts=artifacts,
            config=replace(
                load_evaluation_config(), categories=categories, image_size=32, warmup_images=1
            ),
            output=tmp_path / "evaluation",
        )

    return make


def _run(prepared, **overrides):
    values = dict(data_kind="synthetic", manifest_dir=prepared.manifests, config=prepared.config)
    values.update(overrides)
    return evaluate(prepared.root, prepared.artifacts, prepared.output, **values)


def test_real_models_freeze_before_any_test_read_and_publish_metrics(prepare, monkeypatch):
    prepared = prepare()
    original_read, original_audit = experiment._read, experiment.inspect_category
    reads = []

    def assert_frozen():
        calibration = json.loads((prepared.output / "calibration.json").read_text())
        assert calibration["complete"] is False
        assert set(calibration["calibration"]["bottle"]) == {"autoencoder", "patchcore"}

    def guarded_read(root, image_id, **kwargs):
        if "/test/" in image_id or "/ground_truth/" in image_id:
            assert_frozen()
        reads.append(image_id)
        return original_read(root, image_id, **kwargs)

    def guarded_audit(*args):
        assert_frozen()
        return original_audit(*args)

    monkeypatch.setattr(experiment, "_read", guarded_read)
    monkeypatch.setattr(experiment, "inspect_category", guarded_audit)
    path = _run(prepared)
    report = json.loads(path.read_text())
    assert report["complete"] is True
    assert report["data_kind"] == "synthetic" and report["protocol_scope"] == "partial"
    assert report["config_hash"] == prepared.config.config_hash
    assert len(report["results"]) == 2
    first, second = report["results"]
    assert first["test_ids"] == second["test_ids"] == report["tests"]["bottle"]["test_ids"]
    assert len(first["test_ids"]) == 2
    assert not set(prepared.splits["bottle"].train_ids) & set(reads)
    for result in report["results"]:
        assert result["thresholds"]["image"]["sample_count"] == 2
        assert result["thresholds"]["pixel"]["sample_count"] == 2048
        assert result["thresholds"]["image"]["source"] == "normal_validation"
        assert result["compute"]["inference_latency_ms"] > 0
        assert result["compute"]["throughput_images_per_s"] > 0
        assert result["compute"]["prepare_time_s"] > 0
        assert result["compute"]["artifact_size_bytes"] > 0
        assert "not_model_isolated" in result["compute"]["peak_memory_source"]
        assert result["metrics"]["pixel_auroc"] is not None
    with (prepared.output / "results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["threshold_source"] == "normal_validation" for row in rows)
    assert report["csv_sha256"] == experiment.file_checksum(prepared.output / "results.csv")
    assert not list(prepared.output.glob("*.tmp"))


def test_all_three_categories_freeze_before_first_test_audit(prepare, monkeypatch):
    prepared = prepare(("metal_nut", "bottle", "leather"))
    original = experiment.inspect_category

    def guarded(*args):
        calibration = json.loads((prepared.output / "calibration.json").read_text())
        assert set(calibration["calibration"]) == set(prepared.config.categories)
        assert all(len(models) == 2 for models in calibration["calibration"].values())
        return original(*args)

    monkeypatch.setattr(experiment, "inspect_category", guarded)
    result = json.loads(_run(prepared).read_text())
    assert len(result["results"]) == 6
    assert result["protocol_scope"] == "complete"


def test_changed_official_test_values_and_masks_cannot_change_thresholds(prepare):
    prepared = prepare()
    original = json.loads(_run(prepared).read_text())
    _rgb(prepared.root / "bottle/test/broken/000.png", 210)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    Image.fromarray(mask).save(prepared.root / "bottle/ground_truth/broken/000_mask.png")
    prepared.output = prepared.output.with_name("second-evaluation")
    changed = json.loads(_run(prepared).read_text())
    assert original["calibration"] == changed["calibration"]
    assert original["tests"] != changed["tests"]


@pytest.mark.parametrize("mismatch", ["split", "seed", "category", "resolution", "grayscale"])
def test_incompatible_artifacts_rejected_before_test(prepare, monkeypatch, mismatch):
    prepared = prepare()
    original = experiment._load_artifact

    def changed(path, model_id, device):
        predictor, metadata, digest = original(path, model_id, device)
        if model_id == "patchcore":
            if mismatch == "split":
                metadata["split"]["train_ids"] = []
            elif mismatch == "seed":
                metadata["config"]["seed"] = 43
            else:
                key, value = {
                    "category": ("category_id", "leather"),
                    "resolution": ("image_size", 64),
                    "grayscale": ("grayscale", True),
                }[mismatch]
                predictor.config = replace(
                    predictor.config,
                    preprocess=replace(predictor.config.preprocess, **{key: value}),
                )
        return predictor, metadata, digest

    monkeypatch.setattr(experiment, "_load_artifact", changed)
    monkeypatch.setattr(experiment, "inspect_category", lambda *_: pytest.fail("Test opened early"))
    with pytest.raises(ValueError):
        _run(prepared)
    assert not (prepared.output / "calibration.json").exists()
    assert not (prepared.output / "results.json").exists()


def test_random_weights_not_allowed_as_real_experiment(prepare, monkeypatch):
    prepared = prepare()
    monkeypatch.setattr(experiment, "inspect_category", lambda *_: pytest.fail("Test opened early"))
    with pytest.raises(ValueError, match="pretrained"):
        _run(prepared, data_kind="mvtec_ad")


def test_corrupt_test_fails_only_after_calibration_and_never_publishes_result(prepare):
    prepared = prepare()
    (prepared.root / "bottle/test/good/000.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        _run(prepared)
    assert (prepared.output / "calibration.json").exists()
    assert not (prepared.output / "results.json").exists()


def test_pixel_budget_fails_without_test_peek(prepare, monkeypatch):
    prepared = prepare()
    monkeypatch.setattr(experiment, "inspect_category", lambda *_: pytest.fail("Test opened early"))
    with pytest.raises(ValueError, match="budget"):
        _run(prepared, config=replace(prepared.config, max_pixels=1))


def test_test_pixel_budget_is_checked_after_freeze(prepare):
    prepared = prepare()
    _rgb(prepared.root / "bottle/test/good/001.png", 130)
    with pytest.raises(ValueError, match="budget"):
        _run(prepared, config=replace(prepared.config, max_pixels=2500))
    assert (prepared.output / "calibration.json").exists()


def test_missing_history_records_unknown_not_invented_duration(prepare):
    prepared = prepare()
    prepared.artifacts["bottle"]["autoencoder"].with_name("history.json").unlink()
    result = json.loads(_run(prepared).read_text())
    compute = result["results"][0]["compute"]
    assert compute["prepare_time_s"] is None
    assert compute["prepare_time_source"] == "unknown_missing_history_json"


def test_history_must_be_bound_to_checkpoint_and_duration_valid(prepare):
    prepared = prepare()
    history = prepared.artifacts["bottle"]["autoencoder"].with_name("history.json")
    record = json.loads(history.read_text())
    record["checkpoint_sha256"] = "0" * 64
    history.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="bound"):
        _run(prepared)
    assert not (prepared.output / "calibration.json").exists()


def test_invalid_history_duration_fails_explicitly(prepare):
    prepared = prepare()
    history = prepared.artifacts["bottle"]["autoencoder"].with_name("history.json")
    record = json.loads(history.read_text())
    record["duration_seconds"] = float("nan")
    history.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="duration"):
        _run(prepared)


def test_history_drift_after_freeze_fails(prepare, monkeypatch):
    prepared = prepare()
    original = experiment._json

    def changed(path, value):
        original(path, value)
        if path.name == "calibration.json":
            history = prepared.artifacts["bottle"]["autoencoder"].with_name("history.json")
            record = json.loads(history.read_text())
            record["duration_seconds"] += 1
            history.write_text(json.dumps(record))

    monkeypatch.setattr(experiment, "_json", changed)
    with pytest.raises(ValueError, match="history changed"):
        _run(prepared)


def test_artifact_drift_after_freeze_rejected(prepare, monkeypatch):
    prepared = prepare()
    original = experiment._json

    def publish(path, value):
        original(path, value)
        if path.name == "calibration.json":
            checkpoint = prepared.artifacts["bottle"]["autoencoder"]
            payload = torch.load(checkpoint, weights_only=True)
            payload["metadata"]["extra"] = "changed after freeze"
            torch.save(payload, checkpoint)

    monkeypatch.setattr(experiment, "_json", publish)
    with pytest.raises(ValueError, match="Artifact changed"):
        _run(prepared)
    assert not (prepared.output / "results.json").exists()


def test_test_source_drift_between_model_reads_rejected(prepare, monkeypatch):
    prepared = prepare()
    original = experiment._read
    count = 0

    def changed(root, image_id, **kwargs):
        nonlocal count
        if image_id == "bottle/test/good/000.png":
            count += 1
            if count == 3:  # Inventory read, Autoencoder read, PatchCore read.
                _rgb(root / image_id, 120)
        return original(root, image_id, **kwargs)

    monkeypatch.setattr(experiment, "_read", changed)
    with pytest.raises(ValueError, match="Test image changed"):
        _run(prepared)


def test_failed_completed_publication_leaves_only_incomplete_run(prepare, monkeypatch):
    prepared = prepare()
    original = experiment._write

    def failing(path, content):
        if path.name == "results.json":
            raise OSError("disk unavailable")
        return original(path, content)

    monkeypatch.setattr(experiment, "_write", failing)
    with pytest.raises(OSError, match="disk unavailable"):
        _run(prepared)
    assert (prepared.output / "calibration.json").exists()
    assert (prepared.output / "results.csv").exists()
    assert not (prepared.output / "results.json").exists()
    assert not list(prepared.output.glob("*.tmp"))


def test_existing_run_invalid_artifact_mapping_and_dataset_output_rejected(prepare):
    prepared = prepare()
    prepared.output.mkdir()
    with pytest.raises(FileExistsError):
        _run(prepared)
    prepared.output = prepared.root / "report"
    with pytest.raises(ValueError, match="outside"):
        _run(prepared)
    assert not prepared.output.exists()
    prepared.output = prepared.root.parent / "unused"
    prepared.artifacts["bottle"].pop("patchcore")
    with pytest.raises(ValueError, match="both model artifacts"):
        _run(prepared)
    assert not prepared.output.exists()


def test_cli_real_models_success_and_duplicate_artifact_failure(prepare, tmp_path):
    prepared = prepare()
    protocol = tmp_path / "protocol.yaml"
    import yaml

    protocol.write_text(
        yaml.safe_dump(
            {
                **prepared.config.to_dict(),
                "split_policy": "validation thresholds, then a single official test report",
            }
        )
    )
    command = [
        sys.executable,
        "scripts/evaluate_experiment.py",
        "--protocol",
        str(protocol),
        "--dataset-root",
        str(prepared.root),
        "--manifest-dir",
        str(prepared.manifests),
        "--output-dir",
        str(prepared.output),
        "--data-kind",
        "synthetic",
        "--autoencoder",
        f"bottle={prepared.artifacts['bottle']['autoencoder']}",
        "--patchcore",
        f"bottle={prepared.artifacts['bottle']['patchcore']}",
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)["results"]).exists()
    result = subprocess.run(
        command + ["--autoencoder", "bottle=duplicate.pt"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 1
    assert "exactly one" in result.stderr
