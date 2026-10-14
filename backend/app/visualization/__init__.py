"""Heatmap and overlay encoding.

Wave 6 converts anomaly maps to PNG. Wave 0 reserves the package.
"""


def encode_anomaly_map(*_args: object, **_kwargs: object) -> None:
    """Encode a float anomaly map as PNG. Not available in Wave 0."""
    raise NotImplementedError("Anomaly map rendering is implemented in Wave 6.")
