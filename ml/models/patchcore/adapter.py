"""Frozen Anomalib PatchCore backend and shared project prediction contract.

HWC RGB uint8 [0,255] enters the recorded ImageNet-normalized CHW float32 path.
Anomalib provides a square HW float32 nonnegative unbounded feature-distance map
and its raw weighted maximum-patch score (plain maximum for neighbors=1). The
backbone never receives gradients or batchnorm updates. No labels or thresholds.
"""

import random
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from ml.models import AnomalyPrediction
from ml.models.autoencoder.prediction import resolve_device
from ml.preprocessing import apply_preprocess

from .config import PatchcoreConfig


@contextmanager
def seeded_backend(seed: int):
    """Restore caller torch/NumPy/Python RNG and deterministic backend flags."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    dtype = torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, cudnn_deterministic = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            torch.manual_seed(seed)
            # Anomalib's sparse projection allocates some tensors from defaults.
            torch.set_default_dtype(torch.float32)
            np.random.seed(seed)
            random.seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            torch.set_default_dtype(dtype)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = (
                benchmark,
                cudnn_deterministic,
            )


def new_backend(
    config: PatchcoreConfig, *, device: str = "cpu", reload: bool = False
) -> PatchcoreModel:
    target = resolve_device(device)
    with seeded_backend(config.seed):
        model = PatchcoreModel(
            backbone=config.backbone,
            layers=list(config.layers),
            pre_trained=False if reload else config.pre_trained,
            num_neighbors=config.num_neighbors,
        ).to(device=target, dtype=torch.float32)
    model.requires_grad_(False)
    model.eval()
    return model


def select_memory(model: PatchcoreModel, config: PatchcoreConfig) -> None:
    # Anomalib 2.7 warns about its omitted optional embeddings argument too.
    # The adapter never passes that deprecated argument; retain the supported API.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="Argument embeddings is deprecated since v2.1.0.",
        )
        model.subsample_embedding(config.coreset_sampling_ratio)


@dataclass(frozen=True)
class PatchcorePrediction(AnomalyPrediction):
    input_rgb: np.ndarray
    preprocessing_record: dict


class PatchcorePredictor:
    def __init__(
        self, model: PatchcoreModel, config: PatchcoreConfig, *, device: str = "cpu"
    ) -> None:
        if not isinstance(config, PatchcoreConfig) or not isinstance(model, PatchcoreModel):
            raise ValueError("Prediction requires a configured Anomalib PatchcoreModel.")
        if (
            model.backbone != config.backbone
            or tuple(model.layers) != config.layers
            or model.num_neighbors != config.num_neighbors
        ):
            raise ValueError("PatchCore backend and configuration disagree.")
        memory = model.memory_bank
        if (
            memory.ndim != 2
            or memory.shape[0] < config.num_neighbors
            or memory.shape[1] != sum(model.feature_extractor.out_dims)
            or memory.dtype != torch.float32
            or not torch.isfinite(memory).all()
        ):
            raise ValueError("Prepared PatchCore memory is empty, too small or invalid.")
        self.device = resolve_device(device)
        self.model = model.to(device=self.device, dtype=torch.float32).requires_grad_(False).eval()
        self.config = config

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> PatchcorePrediction:
        prepared = apply_preprocess(image, self.config.preprocess, stage="inference")
        tensor = torch.from_numpy(prepared.model_input).unsqueeze(0).to(self.device)
        self.model.eval()
        output = self.model(tensor)
        anomaly_map = output.anomaly_map[0, 0].cpu().numpy().astype(np.float32, copy=True)
        score = float(output.pred_score[0].cpu())
        size = self.config.preprocess.image_size
        if (
            anomaly_map.shape != (size, size)
            or not np.isfinite(anomaly_map).all()
            or (anomaly_map < 0).any()
            or not np.isfinite(score)
            or score < 0
        ):
            raise ValueError("Anomalib returned an invalid raw score or map.")
        return PatchcorePrediction(score, anomaly_map, prepared.display_rgb, prepared.record)
