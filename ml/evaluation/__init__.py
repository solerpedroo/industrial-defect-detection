"""Metric names shared by evaluation scripts and later API exports.

Functions that compute the metrics arrive in Wave 5. Names are stable so tests
can lock the contract early.
"""

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


def evaluate(*_args: object, **_kwargs: object) -> None:
    """Evaluate a prepared model. Not available in Wave 0."""
    raise NotImplementedError("Evaluation metrics are implemented in Wave 5.")
