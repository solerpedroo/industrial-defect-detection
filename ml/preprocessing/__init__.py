"""Recorded preprocessing settings.

Wave 2 turns ``PreprocessConfig`` into the resize and normalization path shared
by training and inference. Augmentations stay opt-in per category.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    """Spatial size and RGB normalization recorded with each experiment."""

    image_size: int = 256
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    grayscale: bool = False


def apply_preprocess(*_args: object, **_kwargs: object) -> None:
    """Resize and normalize an image. Not available in Wave 0."""
    raise NotImplementedError("Preprocessing is implemented in Wave 2.")
