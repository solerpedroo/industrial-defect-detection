"""Offline normal-only Adam/MSE fitting with best-validation checkpoints.

Only normal train/good IDs from a valid manifest are read. Shared preprocessing
produces CHW float32 RGB [0,1]; masks and official test pixels are never opened.
Prepared artifacts stay outside Git. No thresholds, binary labels or HTTP calls.
"""

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader, Dataset

from ml.datasets import load_split_manifest
from ml.datasets.reader import _open_regular_file, file_checksum
from ml.models.autoencoder.checkpoint import save_checkpoint, save_history
from ml.models.autoencoder.network import ConvolutionalAutoencoder
from ml.models.autoencoder.prediction import resolve_device
from ml.preprocessing import apply_preprocess
from ml.training.config import AutoencoderTrainConfig, load_autoencoder_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def sample_seed(seed: int, epoch: int, image_id: str) -> int:
    """SHA-256(seed:epoch:image_id), first four bytes interpreted big-endian."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{image_id}".encode()).digest()[:4], "big")


class _NormalDataset(Dataset):
    def __init__(
        self,
        root: Path,
        ids: tuple[str, ...],
        checksums: dict[str, str],
        config: AutoencoderTrainConfig,
        stage: str,
    ) -> None:
        self.root, self.ids, self.checksums = root, ids, dict(checksums)
        self.config, self.stage, self.epoch = config, stage, 0

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_id = self.ids[index]
        path = self.root / image_id
        if any(parent.is_symlink() for parent in path.parents):
            raise ValueError("Normal image path contains a symlink.")
        with _open_regular_file(path) as stream:
            raw = stream.read()
        if hashlib.sha256(raw).hexdigest() != self.checksums[image_id]:
            raise ValueError(f"Normal training image changed: {image_id}.")
        with Image.open(BytesIO(raw)) as source:
            source.load()
            pixels = np.ascontiguousarray(np.asarray(source.convert("RGB"), dtype=np.uint8))
        prepared = apply_preprocess(
            pixels,
            self.config.preprocess,
            stage=self.stage,
            seed=sample_seed(self.config.seed, self.epoch, image_id)
            if self.stage == "train"
            else None,
        )
        return torch.from_numpy(prepared.model_input)


@dataclass
class EarlyStopping:
    patience: int
    best_loss: float = math.inf
    best_epoch: int = 0
    stale_epochs: int = 0

    def __post_init__(self) -> None:
        if type(self.patience) is not int or self.patience <= 0:
            raise ValueError("Early stopping patience must be a positive integer.")

    def observe(self, loss: float, epoch: int) -> bool:
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("Epoch must be a positive integer.")
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("Validation loss must be finite and nonnegative.")
        improved = loss < self.best_loss
        if improved:
            self.best_loss, self.best_epoch, self.stale_epochs = loss, epoch, 0
        else:
            self.stale_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.stale_epochs >= self.patience


@contextmanager
def _reproducible_training(seed: int):
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, cudnn_deterministic = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    with torch.random.fork_rng(devices=devices):
        try:
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            yield
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.deterministic = cudnn_deterministic


def _epoch_loss(model, loader, device, optimizer=None) -> float:
    total, count = 0.0, 0
    model.train(optimizer is not None)
    for batch in loader:
        batch = batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("Training produced a non-finite loss.")
            loss.backward()
            optimizer.step()
        else:
            with torch.inference_mode():
                loss = torch.nn.functional.mse_loss(model(batch), batch)
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise ValueError("Training or validation produced a non-finite loss.")
        total += value * batch.shape[0]
        count += batch.shape[0]
    if not count:
        raise ValueError("Normal train and validation loaders must be nonempty.")
    return total / count


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    history_path: Path
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int


def train_autoencoder(
    dataset_root: Path,
    category_id: str,
    output_dir: Path,
    *,
    manifest_dir: Path | None = None,
    config: AutoencoderTrainConfig | None = None,
    device: str = "cpu",
) -> TrainingResult:
    """Fit normal manifest IDs using shared RGB [0,1] preprocessing and Adam/MSE.

    Requires a new owned output directory outside source data and outside tracked
    repository roots. Checkpoints/history contain normal split IDs and checksums,
    effective config/seed/versions/device and sample-weighted epoch losses. The
    saved state is the strict best normal-validation epoch, never test-selected.
    """
    root, output = Path(dataset_root), Path(output_dir).resolve()
    canonical = load_autoencoder_config(category_id)
    config = canonical if config is None else config
    if (
        not isinstance(config, AutoencoderTrainConfig)
        or config.preprocess.category_id != category_id
    ):
        raise ValueError("Training config must match the requested catalog category.")
    if config.preprocess.allow_horizontal_flip and not canonical.preprocess.allow_horizontal_flip:
        raise ValueError("Training cannot override the canonical category flip permission.")
    resolved_device = resolve_device(device)
    if output.is_relative_to(root.resolve()):
        raise ValueError("Output directory must be outside the source dataset.")
    if output.is_relative_to(REPO_ROOT) and not any(
        output.is_relative_to(REPO_ROOT / name) and output != REPO_ROOT / name
        for name in ("checkpoints", "artifacts", "runs")
    ):
        raise ValueError("Repository runs must live beneath checkpoints/, artifacts/ or runs/.")
    manifest = load_split_manifest(root, category_id, manifest_dir=manifest_dir)
    train = _NormalDataset(root, manifest.train_ids, manifest.train_checksums, config, "train")
    validation = _NormalDataset(
        root, manifest.validation_ids, manifest.train_checksums, config, "validation"
    )
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path, history_path = output / "best.pt", output / "history.json"
    metadata = {
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "preprocessing_config_hash": config.preprocess.config_hash,
        "split": json.loads(json.dumps(manifest.to_dict())),
        "seed_derivation": "sha256(seed:epoch:image_id) first four bytes big-endian",
        "device": str(resolved_device),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "device_name": torch.cuda.get_device_name(resolved_device)
            if resolved_device.type == "cuda"
            else "CPU",
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
    }
    history: list[dict] = []
    best_state: dict[str, torch.Tensor] | None = None
    stopping = EarlyStopping(config.early_stopping_patience)
    started = time.perf_counter()
    with _reproducible_training(config.seed):
        model = ConvolutionalAutoencoder(config.channels).to(resolved_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        train_loader = DataLoader(
            train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(config.seed),
        )
        validation_loader = DataLoader(
            validation,
            batch_size=config.batch_size,
            num_workers=0,
            generator=torch.Generator().manual_seed(config.seed + 1),
        )
        for epoch in range(1, config.max_epochs + 1):
            train.epoch = epoch
            train_loss = _epoch_loss(model, train_loader, resolved_device, optimizer)
            validation_loss = _epoch_loss(model, validation_loader, resolved_device)
            improved = stopping.observe(validation_loss, epoch)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "improved": improved,
                }
            )
            if improved:
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                save_checkpoint(
                    checkpoint_path,
                    model,
                    {
                        **metadata,
                        "complete": False,
                        "best_epoch": epoch,
                        "best_validation_loss": validation_loss,
                    },
                )
            save_history(
                history_path,
                {
                    **metadata,
                    "history": history,
                    "complete": False,
                    "best_epoch": stopping.best_epoch,
                    "best_validation_loss": stopping.best_loss,
                },
            )
            if stopping.should_stop:
                break
    if load_split_manifest(root, category_id, manifest_dir=manifest_dir) != manifest:
        raise ValueError("Normal split changed during training; run remains incomplete.")
    model.load_state_dict(best_state, strict=True)
    with tempfile.NamedTemporaryFile(
        dir=output, prefix=".complete-", suffix=".tmp", delete=False
    ) as stream:
        staged = Path(stream.name)
    final_history = {
        **metadata,
        "history": history,
        "complete": True,
        "best_epoch": stopping.best_epoch,
        "best_validation_loss": stopping.best_loss,
        "duration_seconds": time.perf_counter() - started,
        "stopped_early": stopping.should_stop,
    }
    try:
        save_checkpoint(
            staged,
            model,
            {
                **metadata,
                "complete": True,
                "best_epoch": stopping.best_epoch,
                "best_validation_loss": stopping.best_loss,
            },
        )
        final_history["checkpoint_sha256"] = file_checksum(staged)
        save_history(history_path, final_history)
        # The completed checkpoint is the last publication and the readiness marker.
        os.replace(staged, checkpoint_path)
    except OSError:
        # Best weights remain incomplete if final publication fails. Where the
        # filesystem still permits writes, keep the history flag consistent too.
        try:
            save_history(history_path, {**final_history, "complete": False})
        except OSError:
            pass
        raise
    finally:
        staged.unlink(missing_ok=True)
    return TrainingResult(
        checkpoint_path, history_path, stopping.best_epoch, stopping.best_loss, len(history)
    )
