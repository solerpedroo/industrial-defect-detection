"""Score experiment B on copies of the same images at frozen thresholds.

Thresholds come from a completed experiment A report. This module does not
fit a model and does not call calibration. Degraded pixels exist only in
memory. Anomaly scores remain algorithm outputs, not physical severity.
"""

import json
from pathlib import Path

import numpy as np

from ml.evaluation.metrics import evaluate_predictions
from ml.experiments.degrade import apply_degradation
from ml.experiments.protocols import degradation_conditions


def score_degraded_split(
    records: list[dict],
    predict,
    *,
    image_threshold: float,
    pixel_threshold: float,
    seed: int,
    fpr_limit: float = 0.3,
    conditions: tuple[tuple[str, float | None], ...] | None = None,
) -> list[dict]:
    """Evaluate every condition with the supplied frozen thresholds.

    ``records`` items provide ``image_id``, RGB ``uint8`` ``image``, boolean
    ``label`` and a boolean mask the same height and width as ``image``.
    ``predict`` returns ``(score, map)`` with a float32 map of that spatial size.
    """
    if not records:
        raise ValueError("Robustness scoring requires at least one base image.")
    image_ids = [record["image_id"] for record in records]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Base image ids must be unique across conditions.")
    chosen = degradation_conditions() if conditions is None else conditions
    rows = []
    for kind, intensity in chosen:
        labels, scores, masks, maps = [], [], [], []
        for record in records:
            source = np.asarray(record["image"])
            before = source.copy()
            degraded = apply_degradation(
                source, kind, intensity, seed=seed, image_id=record["image_id"]
            )
            if not np.array_equal(source, before):
                raise ValueError("Degradation modified the base image.")
            score, anomaly_map = predict(degraded)
            anomaly_map = np.asarray(anomaly_map)
            mask = np.asarray(record["mask"], dtype=bool)
            if anomaly_map.shape != mask.shape or anomaly_map.dtype != np.float32:
                raise ValueError("Predictor map must be float32 with the mask shape.")
            if not np.isfinite(anomaly_map).all() or (anomaly_map < 0).any():
                raise ValueError("Anomaly map must be finite and nonnegative.")
            if not np.isfinite(score) or float(score) < 0:
                raise ValueError("Anomaly score must be finite and nonnegative.")
            labels.append(bool(record["label"]))
            scores.append(float(score))
            masks.append(mask)
            maps.append(anomaly_map)
        metrics = evaluate_predictions(
            labels,
            scores,
            np.stack(masks),
            np.stack(maps),
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
            fpr_limit=fpr_limit,
        )
        rows.append(
            {
                "condition": kind,
                "intensity": intensity,
                "test_ids": list(image_ids),
                "image_threshold": image_threshold,
                "pixel_threshold": pixel_threshold,
                "threshold_source": "frozen_normal_validation",
                "metrics": metrics,
            }
        )
    if any(row["test_ids"] != image_ids for row in rows):
        raise ValueError("Conditions must keep the same base image ids.")
    return rows


def thresholds_from_experiment_a(document: dict, category: str, model_id: str) -> tuple[float, float]:
    """Read frozen image and pixel thresholds. Do not recompute them."""
    if document.get("complete") is not True or document.get("phase") != "official_test_report":
        raise ValueError("Robustness requires a completed experiment A report.")
    if document.get("config", {}).get("protocol_id") != "exp-a-autoencoder-vs-patchcore":
        raise ValueError("Thresholds must come from experiment A.")
    matches = [
        row
        for row in document.get("results", [])
        if row.get("category") == category and row.get("model_id") == model_id
    ]
    if len(matches) != 1:
        raise ValueError("Experiment A report must contain one row for this category and model.")
    image = matches[0]["thresholds"]["image"]["threshold"]
    pixel = matches[0]["thresholds"]["pixel"]["threshold"]
    source = matches[0]["thresholds"]["image"].get("source")
    if source != "normal_validation":
        raise ValueError("Experiment A thresholds must come from normal validation.")
    return float(image), float(pixel)


def load_experiment_a(path: Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("Experiment A report must be a JSON object.")
    return document
