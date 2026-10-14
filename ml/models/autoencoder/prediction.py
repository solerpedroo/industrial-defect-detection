"""Prepared autoencoder prediction, using the shared recorded RGB pipeline."""

from dataclasses import dataclass

import numpy as np
import torch

from ml.models import AnomalyPrediction
from ml.preprocessing import PreprocessConfig, apply_preprocess

from .network import ConvolutionalAutoencoder, reconstruction_error


def resolve_device(name: str) -> torch.device:
    try:
        device = torch.device(name)
    except (TypeError, RuntimeError) as error:
        raise ValueError("Device must be cpu or an available cuda device.") from error
    if device.type not in {"cpu", "cuda"} or (device.type == "cpu" and device.index is not None):
        raise ValueError("Only cpu and cuda devices are supported.")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise ValueError("Requested CUDA device is unavailable.")
    return device


@dataclass(frozen=True)
class ReconstructionPrediction(AnomalyPrediction):
    """Square HW map [0,1], HWC RGB reconstruction [0,1], no binary decision."""

    reconstruction: np.ndarray
    input_rgb: np.ndarray
    preprocessing_record: dict


class AutoencoderPredictor:
    def __init__(
        self, model: ConvolutionalAutoencoder, preprocess: PreprocessConfig, *, device: str = "cpu"
    ) -> None:
        if (
            not isinstance(model, ConvolutionalAutoencoder)
            or not isinstance(preprocess, PreprocessConfig)
            or preprocess.mean != (0, 0, 0)
            or preprocess.std != (1, 1, 1)
            or preprocess.image_size % 8
            or preprocess.image_size > 4096
        ):
            raise ValueError("Prediction requires the unit-interval autoencoder contract.")
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> ReconstructionPrediction:
        """Decode no files; RGB HWC uint8 enters, square float32 map/score exits.

        Error is channel-mean squared RGB difference in [0,1]. The score is the
        map's spatial mean. No threshold/label or original-resolution mask exists.
        """
        prepared = apply_preprocess(image, self.preprocess, stage="inference")
        source = torch.from_numpy(prepared.model_input).unsqueeze(0).to(self.device)
        self.model.eval()
        reconstruction = self.model(source)
        anomaly_map = reconstruction_error(source, reconstruction)[0].cpu().numpy().copy()
        rgb = reconstruction[0].permute(1, 2, 0).cpu().numpy().copy()
        return ReconstructionPrediction(
            score=float(anomaly_map.mean()),
            anomaly_map=anomaly_map,
            reconstruction=rgb,
            input_rgb=prepared.display_rgb,
            preprocessing_record=prepared.record,
        )
