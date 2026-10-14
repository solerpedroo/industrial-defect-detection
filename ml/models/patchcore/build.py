"""Offline normal-only frozen feature extraction and Anomalib coreset construction."""

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ml.datasets import load_split_manifest
from ml.datasets.reader import _open_regular_file, file_checksum
from ml.models.autoencoder.checkpoint import save_history
from ml.preprocessing import apply_preprocess

from .adapter import new_backend, resolve_device, seeded_backend, select_memory
from .artifact import backbone_hash, save_memory
from .config import PatchcoreConfig, load_patchcore_config

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class MemoryBuildResult:
    checkpoint_path: Path
    metadata_path: Path
    patch_count: int
    memory_count: int


def build_patchcore_memory(
    dataset_root: Path,
    category_id: str,
    output_dir: Path,
    *,
    manifest_dir: Path | None = None,
    config: PatchcoreConfig | None = None,
    device: str = "cpu",
) -> MemoryBuildResult:
    """Build frozen normal-only memory, saving a complete prepared run outside Git.

    Uses only retained normal fit IDs, with exact-byte verification and shared
    unaugmented normalization. Validation/test images/masks are never decoded.
    No optimizer, backbone update, test tuning or threshold selection occurs.
    """
    canonical = load_patchcore_config(category_id)
    config = canonical if config is None else config
    if not isinstance(config, PatchcoreConfig) or config.preprocess.category_id != category_id:
        raise ValueError("Memory config must match the requested catalog category.")
    target = resolve_device(device)
    root, output = Path(dataset_root), Path(output_dir).resolve()
    if output.is_relative_to(root.resolve()):
        raise ValueError("Memory output must be outside the source dataset.")
    if output.is_relative_to(REPO_ROOT) and not any(
        output.is_relative_to(REPO_ROOT / name) and output != REPO_ROOT / name
        for name in ("checkpoints", "artifacts", "runs")
    ):
        raise ValueError("Repository memory runs must live beneath ignored artifact roots.")
    if output.exists():
        raise FileExistsError("Memory run directory already exists.")
    manifest = load_split_manifest(root, category_id, manifest_dir=manifest_dir)
    patches = len(manifest.train_ids) * config.patches_per_image
    if patches > config.max_patches:
        raise ValueError(
            f"Normal patch count {patches} exceeds explicit budget {config.max_patches}."
        )
    expected_rows = int(patches * config.coreset_sampling_ratio)
    if expected_rows < config.num_neighbors:
        raise ValueError(
            f"Coreset has {expected_rows} rows; requires at least {config.num_neighbors}."
        )
    # Reserve this run only after preflight, before feature extraction or retrieval.
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path, metadata_path = output / "memory.pt", output / "metadata.json"
    started = time.perf_counter()
    extracted = 0
    with seeded_backend(config.seed):
        model = new_backend(config, device=str(target))
        original_hash = backbone_hash(model)
        model.train()
        model.feature_extractor.eval()
        for offset in range(0, len(manifest.train_ids), config.batch_size):
            batch = []
            for image_id in manifest.train_ids[offset : offset + config.batch_size]:
                path = root / image_id
                if any(parent.is_symlink() for parent in path.parents):
                    raise ValueError("Normal image path contains a symlink.")
                with _open_regular_file(path) as stream:
                    raw = stream.read()
                if hashlib.sha256(raw).hexdigest() != manifest.train_checksums[image_id]:
                    raise ValueError(f"Normal image changed: {image_id}.")
                with Image.open(BytesIO(raw)) as source:
                    source.load()
                    image = np.ascontiguousarray(np.asarray(source.convert("RGB"), dtype=np.uint8))
                prepared = apply_preprocess(image, config.preprocess, stage="train")
                batch.append(torch.from_numpy(prepared.model_input))
            with torch.no_grad():
                embedding = model(torch.stack(batch).to(next(model.parameters()).device))
            if (
                embedding.dtype != torch.float32
                or embedding.ndim != 2
                or not torch.isfinite(embedding).all()
            ):
                raise ValueError("Anomalib returned invalid normal embeddings.")
            extracted += embedding.shape[0]
            if extracted > config.max_patches:
                raise ValueError("Extracted patches exceeded the configured budget.")
        if extracted != patches:
            raise ValueError("Feature geometry disagrees with the configured patch budget.")
        select_memory(model, config)
        model.eval()
        if backbone_hash(model) != original_hash:
            raise ValueError("Backbone changed during frozen memory construction.")
    if load_split_manifest(root, category_id, manifest_dir=manifest_dir) != manifest:
        raise ValueError("Normal source inventory changed during memory construction.")
    if model.memory_bank.shape[0] != expected_rows or not torch.isfinite(model.memory_bank).all():
        raise ValueError("Anomalib coreset does not match the requested sampling ratio.")
    metadata = {
        "complete": True,
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "preprocessing_config_hash": config.preprocess.config_hash,
        "split": json.loads(json.dumps(manifest.to_dict())),
        "patch_count": patches,
        "memory_count": expected_rows,
        "embedding_width": model.memory_bank.shape[1],
        "backbone_sha256": original_hash,
        "pretrained_source": model.feature_extractor.feature_extractor.pretrained_cfg.get(
            "hf_hub_id"
        )
        if config.pre_trained
        else "explicit-random-regression-mode",
        "device": str(next(model.parameters()).device),
        "hardware": {"machine": platform.machine(), "torch_threads": torch.get_num_threads()},
        "versions": {
            name: version(name)
            for name in ("torch", "torchvision", "anomalib", "timm", "numpy", "pillow")
        },
        "python": sys.version.split()[0],
        "duration_seconds": time.perf_counter() - started,
        "score_definition": "Anomalib raw weighted maximum patch nearest-neighbor distance",
    }
    with tempfile.NamedTemporaryFile(
        dir=output, prefix=".memory-", suffix=".tmp", delete=False
    ) as stream:
        staged = Path(stream.name)
    try:
        save_memory(staged, model, metadata)
        save_history(metadata_path, {**metadata, "artifact_sha256": file_checksum(staged)})
        os.replace(staged, checkpoint_path)
    finally:
        staged.unlink(missing_ok=True)
    return MemoryBuildResult(checkpoint_path, metadata_path, patches, expected_rows)
