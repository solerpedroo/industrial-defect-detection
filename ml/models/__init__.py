"""Shared prediction objects for reconstruction and feature-memory models."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class AnomalyPrediction:
    """Image-level score plus a spatial anomaly map.

    ``score`` is an algorithm output, not a physical severity. ``anomaly_map``
    is a finite nonnegative float32 array of shape ``(height, width)`` at model
    resolution. Autoencoder uses squared RGB reconstruction error and spatial
    mean; PatchCore uses raw feature distances and weighted maximum-patch score.
    Numerical scales differ and must not be compared as calibrated probabilities.
    Binary labels require a threshold chosen on validation data.
    """

    score: float
    anomaly_map: np.ndarray


@runtime_checkable
class AnomalyPredictor(Protocol):
    """Shared offline contract: HWC RGB uint8 in [0,255] -> score plus HW map.

    Map/score are finite and nonnegative, but family-specific, with no shared
    upper bound. No threshold, label, calibration or severity is implied.
    """

    def predict(self, image: np.ndarray) -> AnomalyPrediction: ...
