"""Controlled 8-bit degradations for experiment B.

Transforms return a new RGB uint8 array. The source array is not modified.
Gaussian noise uses a seed derived from the protocol seed, image id and sigma
so the same base file yields the same noisy copy. Blur and brightness are
deterministic without an RNG. These arrays are algorithm inputs, not a
physical measure of defect severity.
"""

import hashlib
import math

import numpy as np

NOISE_SIGMAS = (10.0, 20.0, 30.0)
BLUR_SIGMAS = (1.0, 2.0, 3.0)
BRIGHTNESS_FACTORS = (0.6, 0.8, 1.2)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("Degradation expects a contiguous RGB uint8 image.")
    return np.ascontiguousarray(image)


def _kernel(sigma: float) -> np.ndarray:
    if type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Blur sigma must be a positive finite number.")
    radius = math.ceil(3 * float(sigma))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (axis / float(sigma)) ** 2)
    return kernel / kernel.sum()


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _kernel(sigma)
    radius = (kernel.size - 1) // 2
    channels = []
    for channel in np.moveaxis(image.astype(np.float64), 2, 0):
        padded = np.pad(channel, radius, mode="reflect")
        horizontal = np.stack([np.convolve(row, kernel, mode="valid") for row in padded])
        vertical = np.stack([np.convolve(col, kernel, mode="valid") for col in horizontal.T]).T
        channels.append(vertical)
    merged = np.stack(channels, axis=2)
    return np.clip(np.rint(merged), 0, 255).astype(np.uint8)


def _noise(image: np.ndarray, sigma: float, seed: int, image_id: str) -> np.ndarray:
    if type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Noise sigma must be a positive finite number on the 8-bit scale.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32).")
    digest = hashlib.sha256(f"{seed}:{image_id}:gaussian_noise:{sigma}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))
    noisy = image.astype(np.float64) + rng.normal(0.0, float(sigma), size=image.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def _brightness(image: np.ndarray, factor: float) -> np.ndarray:
    if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0:
        raise ValueError("Brightness factor must be a positive finite number.")
    scaled = image.astype(np.float64) * float(factor)
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def apply_degradation(
    image: np.ndarray,
    kind: str,
    intensity: float | None,
    *,
    seed: int,
    image_id: str,
) -> np.ndarray:
    """Return one degraded copy. ``original`` ignores intensity and copies pixels."""
    source = _as_rgb(image)
    if kind == "original":
        return source.copy()
    if kind == "gaussian_noise":
        return _noise(source, float(intensity), seed, image_id)
    if kind == "gaussian_blur":
        return _blur(source, float(intensity))
    if kind == "brightness":
        return _brightness(source, float(intensity))
    raise ValueError(f"Unknown degradation '{kind}'.")
