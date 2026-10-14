"""Atomic owned-run artifacts and restricted loading under an explicit trusted root."""

import json
import os
import pickle
import tempfile
from pathlib import Path

import torch

from ml.datasets.reader import _open_regular_file
from ml.training.config import AutoencoderTrainConfig

from .network import ConvolutionalAutoencoder
from .prediction import AutoencoderPredictor

CHECKPOINT_VERSION = 1


def _atomic_write(path: Path, payload: dict, *, checkpoint: bool) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            if checkpoint:
                torch.save(payload, stream)
            else:
                stream.write(
                    (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(path: Path, model: ConvolutionalAutoencoder, metadata: dict) -> None:
    """Persist inference weights and primitive provenance; no optimizer resume state."""
    json.dumps(metadata, allow_nan=False)
    _atomic_write(
        path,
        {
            "schema_version": CHECKPOINT_VERSION,
            "model_id": "autoencoder",
            "state_dict": {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            },
            "metadata": metadata,
        },
        checkpoint=True,
    )


def save_history(path: Path, document: dict) -> None:
    _atomic_write(path, document, checkpoint=False)


def load_autoencoder(
    checkpoint: Path, *, trusted_root: Path, device: str = "cpu"
) -> AutoencoderPredictor:
    """Load a local trusted checkpoint, validate weights, return an eval predictor.

    No arbitrary pickle globals, modules or optimizer state are loaded. Paths
    must be regular files inside the explicit trusted directory, with no symlinks.
    CPU deserialization precedes transfer to the requested CPU/CUDA device.
    """
    checkpoint, root = Path(checkpoint), Path(trusted_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("trusted_root must be an existing regular artifact directory.")
    try:
        resolved = checkpoint.resolve(strict=True)
        boundary = root.resolve(strict=True)
        if not resolved.is_relative_to(boundary) or not checkpoint.is_file():
            raise ValueError("Checkpoint must be a regular file inside the trusted root.")
        relative = checkpoint.absolute().relative_to(root.absolute())
        current = root.absolute()
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise ValueError("Checkpoint paths must not contain symlinks.")
        if checkpoint.suffix != ".pt":
            raise ValueError("Prepared autoencoder checkpoints must have the .pt extension.")
        with _open_regular_file(checkpoint) as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "model_id",
            "state_dict",
            "metadata",
        }:
            raise ValueError("Unexpected checkpoint fields.")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != CHECKPOINT_VERSION
        ):
            raise ValueError("Unsupported checkpoint version.")
        if payload["model_id"] != "autoencoder":
            raise ValueError("Checkpoint is not an autoencoder.")
        metadata = payload["metadata"]
        json.dumps(metadata, allow_nan=False)
        if not isinstance(metadata, dict) or metadata.get("complete") is not True:
            raise ValueError("Checkpoint is from an incomplete training run.")
        config = AutoencoderTrainConfig.from_dict(metadata["config"])
        if metadata["config_hash"] != config.config_hash:
            raise ValueError("Checkpoint configuration hash does not match its contents.")
        # Initialization on load must not consume the caller's random state.
        with torch.random.fork_rng(devices=[]):
            model = ConvolutionalAutoencoder(config.channels)
        state = payload["state_dict"]
        expected = model.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("Checkpoint state does not match the architecture.")
        for key, tensor in state.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.layout != torch.strided
                or tensor.shape != expected[key].shape
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("Checkpoint weights have invalid shape, dtype or values.")
        model.load_state_dict(state, strict=True)
        return AutoencoderPredictor(model, config.preprocess, device=device)
    except (
        OSError,
        RuntimeError,
        TypeError,
        KeyError,
        pickle.UnpicklingError,
        EOFError,
        RecursionError,
        OverflowError,
    ) as error:
        raise ValueError("Invalid or unreadable trusted autoencoder checkpoint.") from error
