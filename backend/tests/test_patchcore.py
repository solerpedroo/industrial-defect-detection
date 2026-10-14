"""Wave 4: real Anomalib backend, tiny synthetic normal RGB, CPU seed 42."""

import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from ml.datasets import create_split_manifest
from ml.models import AnomalyPrediction, AnomalyPredictor
from ml.models.autoencoder import AutoencoderPredictor, ConvolutionalAutoencoder
from ml.models.patchcore import (
    PatchcoreConfig,
    PatchcorePredictor,
    build_patchcore_memory,
    load_patchcore,
    load_patchcore_config,
)
from ml.models.patchcore.adapter import new_backend
from ml.training.config import load_autoencoder_config


@pytest.fixture(autouse=True)
def preserve_process_state():
    state, threads, dtype = (
        torch.get_rng_state(),
        torch.get_num_threads(),
        torch.get_default_dtype(),
    )
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)
    torch.set_default_dtype(dtype)
    np.random.set_state(numpy_state)
    random.setstate(python_state)


@pytest.fixture
def config():
    original = load_patchcore_config("bottle")
    return replace(
        original,
        preprocess=replace(original.preprocess, image_size=32),
        pre_trained=False,
        batch_size=2,
        coreset_sampling_ratio=0.5,
    )


def _image(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (value, value + 1, value + 2)).save(path)


@pytest.fixture
def dataset(tmp_path):
    root, manifests, output = tmp_path / "mvtec-ad", tmp_path / "manifests", tmp_path / "run"
    for index in range(6):
        _image(root / "bottle/train/good" / f"{index:03}.png", 40 + index)
    _image(root / "bottle/test/good/000.png", 100)
    _image(root / "bottle/test/broken/000.png", 180)
    mask = root / "bottle/ground_truth/broken/000_mask.png"
    mask.parent.mkdir(parents=True)
    Image.new("L", (32, 32), 255).save(mask)
    manifest = create_split_manifest(
        root, "bottle", seed=42, validation_fraction=0.33, manifest_dir=manifests
    )
    return root, manifests, manifest, output


def _build(dataset, config, output=None):
    root, manifests, _, default = dataset
    return build_patchcore_memory(
        root, "bottle", default if output is None else output, manifest_dir=manifests, config=config
    )


def test_real_anomalib_memory_distinguishes_constant_and_bright_patch(dataset, config):
    result = _build(dataset, config)
    predictor = load_patchcore(result.checkpoint_path, trusted_root=dataset[3])
    normal = np.full((32, 32, 3), [40, 41, 42], dtype=np.uint8)
    corrupted = normal.copy()
    corrupted[8:24, 8:24] = 255
    normal_prediction, corrupted_prediction = (
        predictor.predict(normal),
        predictor.predict(corrupted),
    )
    assert corrupted_prediction.score > normal_prediction.score
    assert corrupted_prediction.anomaly_map.mean() > normal_prediction.anomaly_map.mean()
    assert normal_prediction.anomaly_map.shape == (32, 32)
    assert normal_prediction.anomaly_map.dtype == np.float32
    assert np.isfinite(corrupted_prediction.anomaly_map).all()
    assert (corrupted_prediction.anomaly_map >= 0).all()
    assert isinstance(corrupted_prediction, AnomalyPrediction)
    assert isinstance(predictor, AnomalyPredictor)
    assert not hasattr(corrupted_prediction, "label")


def test_memory_shape_artifacts_and_normal_provenance(dataset, config):
    result = _build(dataset, config)
    metadata = json.loads(result.metadata_path.read_text())
    assert result.patch_count == 64 and result.memory_count == 32
    assert metadata["embedding_width"] == 384
    assert metadata["config_hash"] == config.config_hash
    assert metadata["preprocessing_config_hash"] == config.preprocess.config_hash
    assert metadata["config"]["pre_trained"] is False
    assert metadata["pretrained_source"] == "explicit-random-regression-mode"
    assert metadata["split"]["train_ids"] == list(dataset[2].train_ids)
    assert len(metadata["split"]["validation_ids"]) == 2
    assert metadata["complete"] is True and metadata["duration_seconds"] >= 0
    assert metadata["versions"]["anomalib"] == "2.7.0"
    assert sorted(path.name for path in dataset[3].iterdir()) == ["memory.pt", "metadata.json"]
    from ml.datasets.reader import file_checksum

    assert metadata["artifact_sha256"] == file_checksum(result.checkpoint_path)
    predictor = load_patchcore(result.checkpoint_path, trusted_root=dataset[3])
    assert predictor.model.memory_bank.shape == (32, 384)
    assert all(not parameter.requires_grad for parameter in predictor.model.parameters())
    assert not predictor.model.training and not predictor.model.feature_extractor.training


def test_validation_test_and_masks_never_enter_memory(dataset, config, monkeypatch):
    root, _, manifest, _ = dataset
    sealed = [root / image_id for image_id in manifest.validation_ids]
    sealed += list((root / "bottle/test").rglob("*.png"))
    sealed += list((root / "bottle/ground_truth").rglob("*.png"))
    # Validation bytes are still checked by manifest validation but never decoded.
    for path in sealed[len(manifest.validation_ids) :]:
        path.write_bytes(b"sealed-official-evaluation-data")
    from ml.models.patchcore import build as building

    original = building._open_regular_file
    seen = []

    def guarded(path):
        assert path not in sealed
        seen.append(path.relative_to(root).as_posix())
        return original(path)

    monkeypatch.setattr(building, "_open_regular_file", guarded)
    _build(dataset, config)
    assert set(seen) == set(manifest.train_ids)


def test_backbone_and_batchnorm_state_do_not_change_during_build(dataset, config, monkeypatch):
    from ml.models.patchcore import build as building

    original = building.new_backend
    before, backend = {}, []

    def captured(*args, **kwargs):
        model = original(*args, **kwargs)
        before.update(
            {
                key: value.detach().clone()
                for key, value in model.feature_extractor.state_dict().items()
            }
        )
        backend.append(model)
        return model

    monkeypatch.setattr(building, "new_backend", captured)
    _build(dataset, config)
    model = backend[0]
    assert all(
        torch.equal(value, before[key])
        for key, value in model.feature_extractor.state_dict().items()
    )
    assert all(
        parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()
    )
    assert not model.embedding_store


def test_same_environment_memory_is_reproducible_and_global_rng_is_restored(dataset, config):
    torch_state = torch.get_rng_state().clone()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    first = _build(dataset, config)
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert random.getstate() == python_state
    assert flags == (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    second = _build(dataset, config, dataset[3].with_name("second"))
    a, b = (
        torch.load(result.checkpoint_path, weights_only=True)["state_dict"]
        for result in (first, second)
    )
    assert set(a) == set(b)
    assert all(torch.equal(value, b[key]) for key, value in a.items())


def test_caller_default_float64_is_restored_while_patchcore_stays_float32(dataset, config):
    torch.set_default_dtype(torch.float64)
    result = _build(dataset, config)
    predictor = load_patchcore(result.checkpoint_path, trusted_root=dataset[3])
    assert all(parameter.dtype == torch.float32 for parameter in predictor.model.parameters())
    assert predictor.predict(np.zeros((32, 32, 3), np.uint8)).anomaly_map.dtype == np.float32
    assert torch.get_default_dtype() == torch.float64


def test_reload_and_prediction_do_not_fetch_weights_or_enable_gradients(
    dataset, config, monkeypatch
):
    result = _build(dataset, config)
    import timm

    original = timm.create_model
    requests, modes = [], []

    def guarded(*args, **kwargs):
        requests.append(kwargs.get("pretrained", False))
        assert not kwargs.get("pretrained", False)
        return original(*args, **kwargs)

    monkeypatch.setattr(timm, "create_model", guarded)
    first = load_patchcore(result.checkpoint_path, trusted_root=dataset[3])
    second = load_patchcore(result.checkpoint_path, trusted_root=dataset[3])
    first.model.register_forward_pre_hook(
        lambda *args: modes.append(
            (torch.is_grad_enabled(), torch.is_inference_mode_enabled(), first.model.training)
        )
    )
    pixels = np.full((25, 30, 3), 42, np.uint8)
    a, b = first.predict(pixels), second.predict(pixels)
    assert requests and not any(requests)
    assert modes == [(False, True, False)]
    np.testing.assert_array_equal(a.anomaly_map, b.anomaly_map)
    assert a.score == b.score


def test_anomalib_score_definition_is_maximum_patch_when_one_neighbor(config):
    model = new_backend(replace(config, num_neighbors=1))
    scores = torch.tensor([[2.0, 5.0]], dtype=torch.float32)
    actual = model.compute_anomaly_score(
        scores, torch.zeros_like(scores, dtype=torch.long), torch.zeros(2, 384, dtype=torch.float32)
    )
    assert actual.tolist() == [5.0]  # Not spatial mean 3.5.


def test_common_predictor_contract_accepts_autoencoder_and_patchcore(dataset, config):
    ae = load_autoencoder_config("bottle")
    autoencoder = AutoencoderPredictor(
        ConvolutionalAutoencoder(ae.channels), replace(ae.preprocess, image_size=32)
    )
    result = _build(dataset, config)
    patchcore = load_patchcore(result.checkpoint_path, trusted_root=dataset[3])
    image = np.zeros((32, 32, 3), np.uint8)
    for predictor in (autoencoder, patchcore):
        assert isinstance(predictor, AnomalyPredictor)
        prediction = predictor.predict(image)
        assert isinstance(prediction, AnomalyPrediction)
        assert prediction.anomaly_map.shape == (32, 32)
        assert prediction.anomaly_map.dtype == np.float32 and np.isfinite(prediction.score)


def test_pretrained_resnet_smoke_builds_memory_and_reload_is_offline(dataset, config, monkeypatch):
    if os.environ.get("PATCHCORE_PRETRAINED_SMOKE") != "1":
        pytest.skip("Set PATCHCORE_PRETRAINED_SMOKE=1 for the optional pretrained-weight smoke.")
    result = _build(dataset, replace(config, pre_trained=True))
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["config"]["pre_trained"] is True
    assert metadata["pretrained_source"] == "timm/resnet18.a1_in1k"
    import timm

    original = timm.create_model

    def no_fetch(*args, **kwargs):
        assert not kwargs.get("pretrained", False)
        return original(*args, **kwargs)

    monkeypatch.setattr(timm, "create_model", no_fetch)
    predictor = load_patchcore(result.checkpoint_path, trusted_root=dataset[3])
    normal = np.full((32, 32, 3), [40, 41, 42], np.uint8)
    changed = normal.copy()
    changed[8:24, 8:24] = 255
    assert predictor.predict(changed).score > predictor.predict(normal).score


@pytest.mark.parametrize(
    "changes",
    [
        {"backbone": "wide_resnet50_2"},
        {"layers": ("missing",)},
        {"layers": ("layer3", "layer2")},
        {"layers": ("layer2", "layer2")},
        {"pre_trained": "true"},
        {"coreset_sampling_ratio": 0},
        {"coreset_sampling_ratio": float("nan")},
        {"coreset_sampling_ratio": 2},
        {"num_neighbors": 0},
        {"batch_size": True},
        {"max_patches": -1},
        {"seed": -1},
        {"seed": 2**32},
    ],
)
def test_invalid_config_is_rejected(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_canonical_configuration_and_json_round_trip(config):
    path = Path(__file__).resolve().parents[2] / "ml/configs/training_defaults.yaml"
    defaults = yaml.safe_load(path.read_text())
    canonical = load_patchcore_config("bottle")
    assert canonical.pre_trained is True and canonical.backbone == defaults["patchcore"]["backbone"]
    assert canonical.layers == tuple(defaults["patchcore"]["layers"])
    assert canonical.coreset_sampling_ratio == defaults["patchcore"]["coreset_sampling_ratio"]
    assert canonical.preprocess.mean == tuple(defaults["normalization"]["mean"])
    restored = PatchcoreConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config and restored.config_hash == config.config_hash


@pytest.mark.parametrize("changes", [{"coreset_sampling_ratio": 0.01}, {"max_patches": 63}])
def test_insufficient_coreset_or_budget_fails_before_creating_run(dataset, config, changes):
    with pytest.raises(ValueError):
        _build(dataset, replace(config, **changes))
    assert not dataset[3].exists()


def test_invalid_spatial_and_augmented_config_is_rejected(config):
    with pytest.raises(ValueError):
        replace(config, preprocess=replace(config.preprocess, image_size=33))
    augmented = replace(
        config.preprocess, allow_horizontal_flip=True, horizontal_flip_probability=0.5
    )
    with pytest.raises(ValueError):
        replace(config, preprocess=augmented)


def test_existing_and_unsafe_output_paths_are_protected(dataset, config):
    _build(dataset, config)
    before = (dataset[3] / "memory.pt").read_bytes()
    with pytest.raises(FileExistsError):
        _build(dataset, config)
    assert (dataset[3] / "memory.pt").read_bytes() == before
    with pytest.raises(ValueError, match="source dataset"):
        _build(dataset, config, dataset[0] / "run")
    with pytest.raises(ValueError, match="ignored artifact"):
        _build(dataset, config, Path(__file__).resolve().parents[2] / "unsafe-run")


def test_source_drift_during_extraction_cannot_publish_memory(dataset, config, monkeypatch):
    from ml.models.patchcore import build as building

    original = building.apply_preprocess
    changed = False

    def mutate(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            _image(dataset[0] / dataset[2].train_ids[0], 200)
            changed = True
        return result

    monkeypatch.setattr(building, "apply_preprocess", mutate)
    with pytest.raises(ValueError):
        _build(dataset, config)
    assert not (dataset[3] / "memory.pt").exists()


def test_final_metadata_failure_cannot_publish_memory(dataset, config, monkeypatch):
    from ml.models.patchcore import build as building

    def fail(*args, **kwargs):
        raise OSError("Injected metadata failure.")

    monkeypatch.setattr(building, "save_history", fail)
    with pytest.raises(OSError, match="Injected"):
        _build(dataset, config)
    assert not list(dataset[3].iterdir())


@pytest.mark.parametrize(
    "problem", ["version", "model", "incomplete", "hash", "empty", "nan", "backbone"]
)
def test_invalid_memory_artifacts_are_rejected(dataset, config, problem):
    result = _build(dataset, config)
    payload = torch.load(result.checkpoint_path, weights_only=True)
    if problem == "version":
        payload["schema_version"] = True
    elif problem == "model":
        payload["model_id"] = "autoencoder"
    elif problem == "incomplete":
        payload["metadata"]["complete"] = False
    elif problem == "hash":
        payload["metadata"]["config_hash"] = "wrong"
    elif problem == "empty":
        payload["state_dict"]["memory_bank"] = torch.empty(0)
    elif problem == "nan":
        payload["state_dict"]["memory_bank"].fill_(float("nan"))
    else:
        payload["metadata"]["backbone_sha256"] = "wrong"
    torch.save(payload, result.checkpoint_path)
    with pytest.raises(ValueError):
        load_patchcore(result.checkpoint_path, trusted_root=dataset[3])


def test_artifact_paths_reject_escape_symlinks_special_files_and_staging(dataset, config, tmp_path):
    result = _build(dataset, config)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="trusted root"):
        load_patchcore(result.checkpoint_path, trusted_root=other)
    link = dataset[3] / "link.pt"
    link.symlink_to(result.checkpoint_path)
    with pytest.raises(ValueError, match="symlink"):
        load_patchcore(link, trusted_root=dataset[3])
    fifo = dataset[3] / "fifo.pt"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        load_patchcore(fifo, trusted_root=dataset[3])
    staged = dataset[3] / "staged.tmp"
    staged.write_bytes(result.checkpoint_path.read_bytes())
    with pytest.raises(ValueError, match="extension"):
        load_patchcore(staged, trusted_root=dataset[3])


def test_empty_memory_does_not_silently_score(config):
    model = new_backend(config)
    with pytest.raises(ValueError, match="memory"):
        PatchcorePredictor(model, config)


def _cli(dataset, *args):
    root, manifests, _, output = dataset
    repository = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/build_patchcore.py"),
            "--category",
            "bottle",
            "--dataset-root",
            str(root),
            "--manifest-dir",
            str(manifests),
            "--output-dir",
            str(output),
            "--no-pretrained",
            "--image-size",
            "32",
            "--coreset-ratio",
            "0.5",
            *args,
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
        timeout=45,
    )


def test_cli_builds_and_reloads_explicit_regression_memory(dataset):
    result = _cli(dataset)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["pre_trained"] is False and payload["memory_count"] == 32
    assert np.isfinite(
        load_patchcore(Path(payload["checkpoint"]), trusted_root=dataset[3])
        .predict(np.zeros((32, 32, 3), np.uint8))
        .score
    )
    assert _cli(dataset).returncode == 1


@pytest.mark.parametrize(
    "args", [["--coreset-ratio", "0"], ["--image-size", "31"], ["--device", "unknown"]]
)
def test_cli_bad_configuration_creates_no_run(dataset, args):
    result = _cli(dataset, *args)
    assert result.returncode == 1
    assert not dataset[3].exists()
