"""Shared prediction objects for reconstruction and feature-memory models."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnomalyPrediction:
    """Image-level score plus a spatial anomaly map.

    ``score`` is an algorithm output, not a physical severity. ``anomaly_map``
    is a float array of shape ``(height, width)`` with larger values meaning
    more unusual appearance. Binary labels require a threshold chosen on
    validation data.
    """

    score: float
    anomaly_map: np.ndarray
