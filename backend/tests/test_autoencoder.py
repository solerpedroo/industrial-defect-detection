"""Wave 3 CPU regressions: tiny synthetic normal RGB, seed 42, sealed test."""

import json
import os
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
from ml.models.autoencoder import (
    AutoencoderPredictor,
    ConvolutionalAutoencoder,
    load_autoencoder,
    reconstruction_error,
)
from ml.training.autoencoder import EarlyStopping, _epoch_loss, sample_seed, train_autoencoder
from ml.training.config import AutoencoderTrainConfig, load_autoencoder_config


@pytest.fixture(autouse=True)
def preserve_torch_state():
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    dtype = torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)
    torch.set_default_dtype(dtype)
    torch.use_deterministic_algorithms(deterministic)


@pytest.fixture
def config():
    original = load_autoencoder_config("bottle")
    return replace(
        original,
        preprocess=replace(original.preprocess, image_size=8),
        channels=(2, 4, 8),
        batch_size=2,
        max_epochs=4,
        early_stopping_patience=2,
        learning_rate=0.03,
    )


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (value, value + 1, value + 2)).save(path)


@pytest.fixture
def dataset(tmp_path):
    root, manifests = tmp_path / "mvtec-ad", tmp_path / "manifests"
    for index in range(6):
        _image(root / "bottle/train/good" / f"{index:03}.png", 40 + index)
    _image(root / "bottle/test/good/000.png", 100)
    _image(root / "bottle/test/broken/000.png", 180)
    mask = root / "bottle/ground_truth/broken/000_mask.png"
    mask.parent.mkdir(parents=True)
    Image.new("L", (8, 8), 255).save(mask)
    manifest = create_split_manifest(
        root, "bottle", seed=42, validation_fraction=0.33, manifest_dir=manifests
    )
    return root, manifests, manifest, tmp_path / "run"


def _train(dataset, config, output=None):
    root, manifests, _, default_output = dataset
    return train_autoencoder(
        root,
        "bottle",
        default_output if output is None else output,
        manifest_dir=manifests,
        config=config,
    )


@pytest.mark.parametrize("size", [8, 16, 256])
def test_encoder_decoder_restore_size_and_rgb_unit_interval(size):
    torch.manual_seed(42)
    model = ConvolutionalAutoencoder(load_autoencoder_config("bottle").channels)
    batch = torch.full((1, 3, size, size), 0.2, dtype=torch.float32)
    output = model(batch)
    assert output.shape == batch.shape and output.dtype == torch.float32
    assert torch.isfinite(output).all() and (output >= 0).all() and (output <= 1).all()
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_hand_computed_error_map_is_channel_mean_squared_rgb_difference():
    source = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]], dtype=torch.float32)
    result = reconstruction_error(source, torch.zeros_like(source))
    assert result.shape == (1, 1, 1) and result.dtype == torch.float32
    assert float(result[0, 0, 0]) == pytest.approx(1 / 3)
    assert float(reconstruction_error(source, source).sum()) == 0


def test_network_and_training_do_not_inherit_caller_default_dtype(dataset, config):
    torch.set_default_dtype(torch.float64)
    model = ConvolutionalAutoencoder(config.channels)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert model(torch.zeros(1, 3, 8, 8, dtype=torch.float32)).dtype == torch.float32
    result = _train(dataset, config)
    predictor = load_autoencoder(result.checkpoint_path, trusted_root=dataset[3])
    assert predictor.predict(np.zeros((8, 8, 3), np.uint8)).reconstruction.dtype == np.float32
    assert torch.get_default_dtype() == torch.float64


def test_constant_normal_batch_overfits_and_changed_image_outscores_it(config):
    torch.manual_seed(42)
    model = ConvolutionalAutoencoder(config.channels)
    normal = torch.full((2, 3, 8, 8), 0.2, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial = float(torch.nn.functional.mse_loss(model(normal), normal).detach())
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(normal), normal)
        loss.backward()
        optimizer.step()
    final = float(torch.nn.functional.mse_loss(model(normal), normal).detach())
    assert final < initial * 0.2
    predictor = AutoencoderPredictor(model, config.preprocess)
    normal_rgb = np.full((8, 8, 3), 51, dtype=np.uint8)
    changed = normal_rgb.copy()
    changed[2:6, 2:6] = 255
    normal_prediction = predictor.predict(normal_rgb)
    changed_prediction = predictor.predict(changed)
    assert changed_prediction.score > normal_prediction.score
    assert changed_prediction.anomaly_map[2:6, 2:6].mean() > normal_prediction.score
    assert changed_prediction.reconstruction.shape == (8, 8, 3)
    assert changed_prediction.anomaly_map.dtype == np.float32
    assert changed_prediction.score == pytest.approx(float(changed_prediction.anomaly_map.mean()))
    assert not hasattr(changed_prediction, "label")


def test_prediction_uses_eval_and_inference_mode(config):
    model = ConvolutionalAutoencoder(config.channels)
    modes = []
    model.register_forward_pre_hook(
        lambda *args: modes.append(
            (torch.is_grad_enabled(), torch.is_inference_mode_enabled(), model.training)
        )
    )
    result = AutoencoderPredictor(model, config.preprocess).predict(np.zeros((6, 10, 3), np.uint8))
    assert modes == [(False, True, False)]
    assert result.anomaly_map.shape == (8, 8)
    assert result.reconstruction.dtype == np.float32
    assert 0 <= result.score <= 1


def test_training_publishes_history_normal_provenance_and_reloadable_checkpoint(dataset, config):
    result = _train(dataset, config)
    root, _, manifest, output = dataset
    history = json.loads(result.history_path.read_text())
    assert history["complete"] is True
    assert history["history"][-1]["train_loss"] < history["history"][0]["train_loss"]
    assert history["split"]["train_ids"] == list(manifest.train_ids)
    assert history["split"]["validation_ids"] == list(manifest.validation_ids)
    assert set(manifest.train_ids).isdisjoint(manifest.validation_ids)
    assert all("/train/good/" in image_id for image_id in history["split"]["train_checksums"])
    assert history["config_hash"] == config.config_hash
    assert history["preprocessing_config_hash"] == config.preprocess.config_hash
    assert history["versions"]["torch"] == str(torch.__version__)
    assert history["device"] == "cpu" and history["duration_seconds"] >= 0
    assert len(history["checkpoint_sha256"]) == 64
    assert sorted(path.name for path in output.iterdir()) == ["best.pt", "history.json"]
    first = load_autoencoder(result.checkpoint_path, trusted_root=output)
    second = load_autoencoder(result.checkpoint_path, trusted_root=output)
    pixels = np.asarray(Image.open(root / manifest.validation_ids[0]).convert("RGB"))
    np.testing.assert_array_equal(
        first.predict(pixels).reconstruction, second.predict(pixels).reconstruction
    )
    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert checkpoint["metadata"]["best_epoch"] == result.best_epoch
    assert checkpoint["metadata"]["best_validation_loss"] == history["best_validation_loss"]


def test_training_never_opens_test_images_or_masks(dataset, config, monkeypatch):
    root, _, _, _ = dataset
    for path in (root / "bottle/test").rglob("*.png"):
        path.write_bytes(b"official-test-must-stay-sealed")
    for path in (root / "bottle/ground_truth").rglob("*.png"):
        path.write_bytes(b"official-mask-must-stay-sealed")
    from ml.training import autoencoder as training

    open_regular = training._open_regular_file
    seen = []

    def guarded(path):
        if path.is_relative_to(root):
            assert "/train/good/" in path.as_posix()
            seen.append(path.relative_to(root).as_posix())
        return open_regular(path)

    monkeypatch.setattr(training, "_open_regular_file", guarded)
    _train(dataset, config)
    assert set(seen) == set(dataset[2].train_checksums)


def test_repeated_cpu_runs_preserve_rng_and_reproduce_weights(dataset, config):
    state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    first = _train(dataset, config)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    second = _train(dataset, config, dataset[3].with_name("second"))
    a, b = (
        torch.load(result.checkpoint_path, weights_only=True)["state_dict"]
        for result in (first, second)
    )
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert (
        json.loads(first.history_path.read_text())["history"]
        == json.loads(second.history_path.read_text())["history"]
    )
    assert flags == (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])


def test_early_stopping_saves_best_epoch_weights_instead_of_last(dataset, config, monkeypatch):
    from ml.training import autoencoder as training

    original = training._epoch_loss
    losses = iter([0.3, 0.2, 0.25, 0.26])
    snapshots = []

    def controlled(model, loader, device, optimizer=None):
        actual = original(model, loader, device, optimizer)
        if optimizer is not None:
            return actual
        snapshots.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return next(losses)

    monkeypatch.setattr(training, "_epoch_loss", controlled)
    result = _train(dataset, replace(config, max_epochs=10))
    assert result.epochs_completed == 4 and result.best_epoch == 2
    saved = torch.load(result.checkpoint_path, weights_only=True)["state_dict"]
    assert all(torch.equal(value, snapshots[1][key]) for key, value in saved.items())
    assert any(not torch.equal(value, snapshots[-1][key]) for key, value in saved.items())
    assert json.loads(result.history_path.read_text())["stopped_early"] is True


def test_epoch_loss_weights_last_partial_batch_by_sample_count():
    class Zero(torch.nn.Module):
        def forward(self, source):
            return torch.zeros_like(source)

    values = torch.zeros((3, 3, 8, 8), dtype=torch.float32)
    values[-1] = 1
    loader = torch.utils.data.DataLoader(
        values, batch_size=2, generator=torch.Generator().manual_seed(42)
    )
    assert _epoch_loss(Zero(), loader, torch.device("cpu")) == pytest.approx(1 / 3)


def test_seed_derivation_changes_by_sample_and_epoch():
    first = sample_seed(42, 1, "bottle/train/good/000.png")
    assert first == sample_seed(42, 1, "bottle/train/good/000.png")
    assert 0 <= first < 2**32
    assert first != sample_seed(42, 2, "bottle/train/good/000.png")
    assert first != sample_seed(42, 1, "bottle/train/good/001.png")


def test_early_stopping_validates_patience_and_nonfinite_losses():
    with pytest.raises(ValueError):
        EarlyStopping(0)
    stopping = EarlyStopping(2)
    assert stopping.observe(0.3, 1)
    assert not stopping.observe(0.3, 2)
    assert not stopping.should_stop
    assert not stopping.observe(0.4, 3)
    assert stopping.should_stop and stopping.best_epoch == 1
    with pytest.raises(ValueError):
        stopping.observe(float("nan"), 4)


@pytest.mark.parametrize(
    "batch",
    [
        torch.zeros(1, 3, 8, 8, dtype=torch.float64),
        torch.zeros(1, 1, 8, 8),
        torch.zeros(1, 3, 9, 8),
        torch.zeros(0, 3, 8, 8),
        torch.full((1, 3, 8, 8), float("nan")),
        torch.full((1, 3, 8, 8), 2.0),
    ],
)
def test_invalid_model_inputs_fail(batch, config):
    with pytest.raises(ValueError):
        ConvolutionalAutoencoder(config.channels)(batch)


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": 0},
        {"max_epochs": True},
        {"early_stopping_patience": -1},
        {"seed": -1},
        {"seed": 2**32},
        {"learning_rate": float("nan")},
        {"learning_rate": 0},
        {"channels": (2, 4)},
        {"channels": (2, 4, 1000)},
    ],
)
def test_invalid_training_config_fails(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_nonunit_or_incompatible_spatial_preprocessing_is_rejected(config):
    for preprocess in (
        replace(config.preprocess, mean=(0.5, 0.5, 0.5)),
        replace(config.preprocess, image_size=9),
    ):
        with pytest.raises(ValueError):
            replace(config, preprocess=preprocess)


def test_existing_output_and_tracked_repository_paths_are_protected(dataset, config):
    _train(dataset, config)
    before = dataset[3].joinpath("best.pt").read_bytes()
    with pytest.raises(FileExistsError):
        _train(dataset, config)
    assert dataset[3].joinpath("best.pt").read_bytes() == before
    with pytest.raises(ValueError, match="Repository runs"):
        _train(dataset, config, Path(__file__).resolve().parents[2] / "unexpected-run")
    with pytest.raises(ValueError, match="source dataset"):
        _train(dataset, config, dataset[0] / "run")


def test_stale_manifest_and_missing_manifest_fail_before_run_creation(dataset, config):
    root, manifests, manifest, output = dataset
    _image(root / manifest.train_ids[0], 200)
    with pytest.raises(ValueError):
        _train(dataset, config)
    assert not output.exists()
    with pytest.raises(FileNotFoundError):
        train_autoencoder(root, "bottle", output, manifest_dir=manifests / "missing", config=config)
    assert not output.exists()


def test_source_drift_during_fit_leaves_incomplete_checkpoint(dataset, config, monkeypatch):
    from ml.training import autoencoder as training

    original = training._epoch_loss
    root, _, manifest, output = dataset
    changed = False

    def mutate(model, loader, device, optimizer=None):
        nonlocal changed
        value = original(model, loader, device, optimizer)
        if optimizer is not None and not changed:
            _image(root / manifest.train_ids[0], 200)
            changed = True
        return value

    monkeypatch.setattr(training, "_epoch_loss", mutate)
    with pytest.raises(ValueError, match="changed"):
        _train(dataset, config)
    assert json.loads((output / "history.json").read_text())["complete"] is False
    with pytest.raises(ValueError, match="incomplete"):
        load_autoencoder(output / "best.pt", trusted_root=output)


def test_final_history_failure_cannot_publish_completed_checkpoint(dataset, config, monkeypatch):
    from ml.training import autoencoder as training

    original = training.save_history

    def fail_final(path, document):
        if document["complete"]:
            raise OSError("Injected final history write failure.")
        original(path, document)

    monkeypatch.setattr(training, "save_history", fail_final)
    with pytest.raises(OSError, match="Injected"):
        _train(dataset, config)
    output = dataset[3]
    assert json.loads((output / "history.json").read_text())["complete"] is False
    assert sorted(path.name for path in output.iterdir()) == ["best.pt", "history.json"]
    with pytest.raises(ValueError, match="incomplete"):
        load_autoencoder(output / "best.pt", trusted_root=output)


@pytest.mark.parametrize("problem", ["version", "model", "hash", "shape", "nan", "incomplete"])
def test_malformed_checkpoint_is_rejected(dataset, config, problem):
    result = _train(dataset, config)
    payload = torch.load(result.checkpoint_path, weights_only=True)
    if problem == "version":
        payload["schema_version"] = True
    elif problem == "model":
        payload["model_id"] = "patchcore"
    elif problem == "hash":
        payload["metadata"]["config_hash"] = "wrong"
    elif problem in {"shape", "nan"}:
        key = next(iter(payload["state_dict"]))
        if problem == "shape":
            payload["state_dict"][key] = torch.zeros(1)
        else:
            payload["state_dict"][key].fill_(float("nan"))
    else:
        payload["metadata"]["complete"] = False
    torch.save(payload, result.checkpoint_path)
    with pytest.raises(ValueError):
        load_autoencoder(result.checkpoint_path, trusted_root=dataset[3])


def test_checkpoint_path_escape_symlink_special_and_corruption_are_rejected(
    dataset, config, tmp_path
):
    result = _train(dataset, config)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="trusted root"):
        load_autoencoder(result.checkpoint_path, trusted_root=other)
    link = dataset[3] / "link.pt"
    link.symlink_to(result.checkpoint_path)
    with pytest.raises(ValueError, match="symlink"):
        load_autoencoder(link, trusted_root=dataset[3])
    fifo = dataset[3] / "fifo.pt"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        load_autoencoder(fifo, trusted_root=dataset[3])
    result.checkpoint_path.write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError):
        load_autoencoder(result.checkpoint_path, trusted_root=dataset[3])


def test_checkpoint_loading_preserves_caller_rng(dataset, config):
    result = _train(dataset, config)
    state = torch.get_rng_state().clone()
    load_autoencoder(result.checkpoint_path, trusted_root=dataset[3])
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)


def test_loader_always_uses_restricted_cpu_deserialization(dataset, config, monkeypatch):
    result = _train(dataset, config)
    original = torch.load
    seen = []

    def checked(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked)
    load_autoencoder(result.checkpoint_path, trusted_root=dataset[3])
    assert seen == [{"map_location": "cpu", "weights_only": True}]


def test_config_defaults_are_canonical_and_round_trip(config):
    path = Path(__file__).resolve().parents[2] / "ml/configs/training_defaults.yaml"
    defaults = yaml.safe_load(path.read_text())
    loaded = load_autoencoder_config("bottle")
    assert loaded.channels == tuple(defaults["autoencoder"]["channels"])
    assert loaded.batch_size == defaults["autoencoder"]["batch_size"]
    assert loaded.preprocess.mean == (0, 0, 0) and loaded.preprocess.std == (1, 1, 1)
    assert AutoencoderTrainConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_training_cannot_override_category_flip_permission(dataset, config):
    preprocess = replace(
        config.preprocess, allow_horizontal_flip=True, horizontal_flip_probability=1
    )
    with pytest.raises(ValueError, match="canonical category"):
        _train(dataset, replace(config, preprocess=preprocess))


def _cli(dataset, *args):
    root, manifests, _, output = dataset
    repository = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/train_autoencoder.py"),
            "--category",
            "bottle",
            "--dataset-root",
            str(root),
            "--manifest-dir",
            str(manifests),
            "--output-dir",
            str(output),
            *args,
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_cli_trains_reloads_and_refuses_overwrite(dataset):
    result = _cli(dataset, "--epochs", "2", "--image-size", "8", "--batch-size", "2")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["epochs_completed"] == 2
    predictor = load_autoencoder(Path(payload["checkpoint"]), trusted_root=dataset[3])
    assert np.isfinite(predictor.predict(np.zeros((8, 8, 3), np.uint8)).score)
    again = _cli(dataset, "--epochs", "1", "--image-size", "8")
    assert again.returncode == 1 and "File exists" in again.stderr


@pytest.mark.parametrize(
    "args", [["--epochs", "0"], ["--image-size", "9"], ["--device", "unknown"]]
)
def test_cli_invalid_config_writes_no_run(dataset, args):
    result = _cli(dataset, *args)
    assert result.returncode == 1
    assert not dataset[3].exists()
