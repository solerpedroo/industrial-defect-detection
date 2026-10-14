"""Offline metrics/calibration; no optional model library imported here."""

from .calibration import FrozenThreshold, calibrate_normal
from .config import EvaluationConfig, load_evaluation_config
from .metrics import aupro, confusion_metrics, evaluate_predictions, ranking_metrics

__all__ = [
    "FrozenThreshold",
    "EvaluationConfig",
    "calibrate_normal",
    "load_evaluation_config",
    "aupro",
    "confusion_metrics",
    "evaluate_predictions",
    "ranking_metrics",
    "evaluate",
]

IMAGE_METRICS = (
    "image_auroc",
    "image_aupr",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
)

PIXEL_METRICS = (
    "pixel_auroc",
    "pixel_aupr",
    "aupro",
    "iou",
    "dice",
)

COMPUTE_METRICS = (
    "inference_latency_ms",
    "prepare_time_s",
    "artifact_size_bytes",
    "peak_memory_bytes",
    "throughput_images_per_s",
)


def evaluate(*args, **kwargs):
    """Run experiment A offline; model extras load only when this is invoked."""
    from .experiment import run_experiment

    return run_experiment(*args, **kwargs)
