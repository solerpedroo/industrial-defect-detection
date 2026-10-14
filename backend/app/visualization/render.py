"""Model-square PNG transport; calibration-fixed colors are display only."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def png(array: np.ndarray) -> str:
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def render_prediction(prediction, *, size: int, pixel_threshold: float) -> dict:
    """Encode prepared RGB, fixed-range heat, overlay, strict mask, optional reconstruction.

    All images share model-resolution geometry. Color scale [0,2*threshold] never
    adapts to an uploaded image; threshold zero uses binary zero/nonzero intensity.
    Binary mask is uint8 {0,255} from raw map > pixel threshold.
    """
    if (
        type(pixel_threshold) not in (int, float)
        or not np.isfinite(pixel_threshold)
        or not 0 <= pixel_threshold <= np.finfo(np.float32).max
    ):
        raise ValueError("Display requires a finite nonnegative frozen pixel threshold.")
    anomaly_map = np.asarray(prediction.anomaly_map)
    rgb = np.asarray(prediction.input_rgb)
    if (
        anomaly_map.shape != (size, size)
        or anomaly_map.dtype != np.float32
        or not np.isfinite(anomaly_map).all()
        or (anomaly_map < 0).any()
        or rgb.shape != (size, size, 3)
        or rgb.dtype != np.uint8
        or not np.isfinite(prediction.score)
        or prediction.score < 0
    ):
        raise ValueError("Invalid model-square prediction contract.")
    upper = 2 * pixel_threshold
    intensity = (
        np.clip(anomaly_map.astype(np.float64) / upper, 0, 1)
        if upper
        else (anomaly_map > 0).astype(float)
    )
    stops = np.array([[12, 22, 58], [35, 90, 220], [30, 210, 220], [255, 220, 70], [230, 45, 45]])
    heat = np.stack(
        [
            np.interp(intensity, np.linspace(0, 1, len(stops)), stops[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    heat = np.rint(heat).astype(np.uint8)
    result = {
        "input_png": png(rgb),
        "anomaly_map_png": png(heat),
        "overlay_png": png(np.rint(0.5 * rgb + 0.5 * heat).astype(np.uint8)),
        "mask_png": png(((anomaly_map > pixel_threshold) * 255).astype(np.uint8)),
        "reconstruction_png": None,
        "display_max": upper,
    }
    reconstruction = getattr(prediction, "reconstruction", None)
    if reconstruction is not None:
        if (
            reconstruction.shape != rgb.shape
            or not np.isfinite(reconstruction).all()
            or (reconstruction < 0).any()
            or (reconstruction > 1).any()
        ):
            raise ValueError("Invalid reconstruction geometry/range.")
        result["reconstruction_png"] = png(np.rint(reconstruction * 255).astype(np.uint8))
    return result
