"""Prepared PatchCore memory artifacts; restricted CPU loading under a trusted root."""

import hashlib
import json
import pickle
from pathlib import Path

import torch

from ml.datasets.reader import _open_regular_file
from ml.models.autoencoder.checkpoint import _atomic_write

from .adapter import PatchcorePredictor, new_backend
from .config import PatchcoreConfig

ARTIFACT_VERSION = 1


def backbone_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.feature_extractor.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}:".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_memory(path: Path, model, metadata: dict) -> None:
    json.dumps(metadata, allow_nan=False)
    payload = {
        "schema_version": ARTIFACT_VERSION,
        "model_id": "patchcore",
        "metadata": metadata,
        "state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
    }
    _atomic_write(path, payload, checkpoint=True)


def load_patchcore(
    checkpoint: Path, *, trusted_root: Path, device: str = "cpu"
) -> PatchcorePredictor:
    """Load stored backbone and memory without pretrained downloads or arbitrary pickle.

    Only complete regular `.pt` artifacts beneath an explicit trusted root are
    accepted. Deserialize on CPU with weights_only=True, validate schema/config,
    shapes/dtypes/finiteness/coreset size and backbone hash, then freeze and eval.
    """
    checkpoint, root = Path(checkpoint), Path(trusted_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("trusted_root must be an existing regular artifact directory.")
    try:
        if (
            not checkpoint.resolve(strict=True).is_relative_to(root.resolve(strict=True))
            or not checkpoint.is_file()
        ):
            raise ValueError("Memory must be a regular file inside the trusted root.")
        relative = checkpoint.absolute().relative_to(root.absolute())
        current = root.absolute()
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Memory artifact paths must not contain symlinks.")
        if checkpoint.suffix != ".pt":
            raise ValueError("Prepared memory artifacts must use the .pt extension.")
        with _open_regular_file(checkpoint) as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "model_id",
            "metadata",
            "state_dict",
        }:
            raise ValueError("Unexpected memory artifact fields.")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != ARTIFACT_VERSION
        ):
            raise ValueError("Unsupported memory artifact version.")
        if payload["model_id"] != "patchcore":
            raise ValueError("Artifact is not PatchCore.")
        metadata = payload["metadata"]
        json.dumps(metadata, allow_nan=False)
        if not isinstance(metadata, dict) or metadata.get("complete") is not True:
            raise ValueError("Memory build is incomplete.")
        config = PatchcoreConfig.from_dict(metadata["config"])
        if metadata["config_hash"] != config.config_hash:
            raise ValueError("Memory configuration hash is invalid.")
        patches = metadata["patch_count"]
        if type(patches) is not int or not 0 < patches <= config.max_patches:
            raise ValueError("Invalid patch budget provenance.")
        model = new_backend(config, reload=True)
        expected = model.state_dict()
        state = payload["state_dict"]
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("Memory state does not match the configured architecture.")
        rows = int(patches * config.coreset_sampling_ratio)
        width = sum(model.feature_extractor.out_dims)
        for key, value in state.items():
            shape = (rows, width) if key == "memory_bank" else expected[key].shape
            if (
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or value.dtype != expected[key].dtype
                or value.shape != shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Memory weights have invalid shape, dtype or values.")
        if rows < config.num_neighbors or metadata["memory_count"] != rows:
            raise ValueError("Memory does not support the configured neighbor count.")
        model.load_state_dict(state, strict=True)
        if backbone_hash(model) != metadata["backbone_sha256"]:
            raise ValueError("Frozen backbone provenance does not match stored weights.")
        return PatchcorePredictor(model, config, device=device)
    except (
        OSError,
        RuntimeError,
        KeyError,
        TypeError,
        pickle.UnpicklingError,
        EOFError,
        RecursionError,
        OverflowError,
    ) as error:
        raise ValueError("Invalid or unreadable trusted PatchCore memory artifact.") from error
