"""One recorded preprocess path for training, validation, test and inference.

Input: HWC RGB uint8 in [0, 255]. Output: contiguous CHW float32 at the configured
square size, (RGB / 255 - mean) / std, generally outside [0, 1]. Optional grayscale
is replicated to three RGB channels before normalization. No torch import, model
fit, dataset selection or threshold computation occurs here. Training consumers
must use normal fit IDs from a validated Wave 1 manifest.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version

from .config import PREPROCESS_VERSION, PreprocessConfig

Stage = Literal["train", "validation", "test", "inference"]


@dataclass(frozen=True)
class PreprocessResult:
    model_input: np.ndarray
    display_rgb: np.ndarray
    mask: np.ndarray | None
    record: dict


@dataclass(frozen=True)
class PreprocessPreview:
    before_rgb: np.ndarray
    after_rgb: np.ndarray
    before_mask: np.ndarray | None
    after_mask: np.ndarray | None
    record: dict


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input must be a uint8 RGB numpy array in [0, 255].")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError("Input must have nonempty HWC shape (height, width, 3).")


def _validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> None:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8:
        raise ValueError("Mask must be a uint8 array with binary values 0/255.")
    if mask.ndim != 2 or mask.shape != shape or not np.isin(mask, [0, 255]).all():
        raise ValueError("Mask must be binary 0/255 with the source image's HxW shape.")


def apply_preprocess(
    image: np.ndarray,
    config: PreprocessConfig,
    *,
    stage: Stage = "inference",
    mask: np.ndarray | None = None,
    seed: int | None = None,
) -> PreprocessResult:
    """Resize, optionally augment train, then normalize via the same stage path.

    Optional masks are HxW uint8 0/255, resized with nearest-neighbor and flipped
    by the same sampled decision as the image. Returned masks stay uint8 0/255.
    Enabled train flipping requires an explicit seed in [0, 2**32). Non-train
    stages reject seeds and never augment. Inputs and global RNG are unchanged.
    """
    if not isinstance(config, PreprocessConfig):
        raise ValueError("config must be a validated PreprocessConfig.")
    _validate_image(image)
    if not isinstance(stage, str) or stage not in {"train", "validation", "test", "inference"}:
        raise ValueError("Stage must be train, validation, test or inference.")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("Seed must be an integer in [0, 2**32).")
    if stage != "train" and seed is not None:
        raise ValueError("A seed is only accepted for the train stage.")
    augment = stage == "train" and config.horizontal_flip_probability > 0
    if augment and seed is None:
        raise ValueError("Enabled training augmentation requires an explicit seed.")
    if mask is not None:
        _validate_mask(mask, image.shape[:2])

    size = (config.image_size, config.image_size)
    resized = Image.fromarray(image).resize(size, resample=Image.Resampling.BILINEAR)
    paired_mask = (
        np.array(Image.fromarray(mask).resize(size, resample=Image.Resampling.NEAREST))
        if mask is not None
        else None
    )
    flip = (
        bool(np.random.default_rng(seed).random() < config.horizontal_flip_probability)
        if augment
        else False
    )
    if flip:
        resized = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if paired_mask is not None:
            paired_mask = np.ascontiguousarray(paired_mask[:, ::-1])
    if config.grayscale:
        resized = resized.convert("L").convert("RGB")
    display = np.array(resized, dtype=np.uint8, copy=True)
    pixels = display.astype(np.float32) / np.float32(255)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized = (pixels - np.asarray(config.mean, dtype=np.float32)) / np.asarray(
            config.std, dtype=np.float32
        )
    if not np.isfinite(normalized).all():
        raise ValueError("Normalization produced non-finite float32 values.")
    model_input = np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)
    operations = [
        {
            "operation": "resize",
            "source_hw": list(image.shape[:2]),
            "target_hw": list(size),
            "image_resampling": "bilinear",
            "mask_resampling": "nearest" if mask is not None else None,
        },
        {
            "operation": "horizontal_flip",
            "enabled": augment,
            "applied": flip,
            "probability": config.horizontal_flip_probability,
        },
        {
            "operation": "grayscale",
            "applied": config.grayscale,
            "method": "Pillow L",
            "output_channels": 3,
        },
        {"operation": "scale", "divisor": 255},
        {"operation": "normalize", "mean": list(config.mean), "std": list(config.std)},
        {"operation": "layout", "source": "HWC", "target": "CHW"},
    ]
    record = {
        "schema_version": PREPROCESS_VERSION,
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "stage": stage,
        "seed": seed,
        "source_shape": list(image.shape),
        "output_shape": list(model_input.shape),
        "source_dtype": "uint8",
        "output_dtype": "float32",
        "channel_order": "RGB",
        "operations": operations,
        "versions": {"numpy": np.__version__, "pillow": pillow_version},
    }
    return PreprocessResult(model_input, display, paired_mask, record)


def preprocess_preview(
    image: np.ndarray,
    config: PreprocessConfig,
    *,
    stage: Stage = "inference",
    mask: np.ndarray | None = None,
    seed: int | None = None,
) -> PreprocessPreview:
    """Return independent before/after HWC RGB uint8 previews in [0, 255].

    Calls apply_preprocess once, preserving its exact mask geometry and record.
    Preview pixels are pre-normalization display values, not model_input.
    """
    result = apply_preprocess(image, config, stage=stage, mask=mask, seed=seed)
    return PreprocessPreview(
        image.copy(),
        result.display_rgb,
        None if mask is None else mask.copy(),
        result.mask,
        result.record,
    )
