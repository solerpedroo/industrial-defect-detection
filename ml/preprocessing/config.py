"""Validated preprocessing configuration loaded from canonical YAML catalogs."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import yaml

PREPROCESS_VERSION = 1
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _triplet(values: object, name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three RGB channel values.")
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} values must be finite numbers.")
        try:
            number = float(value)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{name} values must be finite float32 numbers.") from error
        if not math.isfinite(number) or abs(number) > float(np.finfo(np.float32).max):
            raise ValueError(f"{name} values must be representable finite float32 numbers.")
        if name == "std" and (number <= 0 or np.float32(number) <= 0):
            raise ValueError("std values must be positive and representable in float32.")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True)
class PreprocessConfig:
    """Effective experiment config; RGB statistics are in unit-interval space.

    Numeric defaults are read by load_preprocess_config, never copied here.
    Direct construction is an explicit experiment configuration, including the
    category's audited flip permission. Use the catalog loader for product paths.
    """

    category_id: str
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    grayscale: bool
    horizontal_flip_probability: float
    allow_horizontal_flip: bool

    def __post_init__(self) -> None:
        if not isinstance(self.category_id, str) or not self.category_id:
            raise ValueError("category_id must be a nonempty catalog identifier.")
        if type(self.image_size) is not int or self.image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        object.__setattr__(self, "mean", _triplet(self.mean, "mean"))
        object.__setattr__(self, "std", _triplet(self.std, "std"))
        if type(self.grayscale) is not bool or type(self.allow_horizontal_flip) is not bool:
            raise ValueError("grayscale and allow_horizontal_flip must be booleans.")
        probability = self.horizontal_flip_probability
        if (
            isinstance(probability, (bool, np.bool_))
            or not isinstance(probability, Real)
            or not 0 <= probability <= 1
        ):
            raise ValueError("horizontal_flip_probability must be a number in [0, 1].")
        object.__setattr__(self, "horizontal_flip_probability", float(probability))
        if probability > 0 and not self.allow_horizontal_flip:
            raise ValueError(f"Horizontal flipping is not permitted for '{self.category_id}'.")

    def to_dict(self) -> dict:
        return {
            "schema_version": PREPROCESS_VERSION,
            **asdict(self),
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _read_yaml(path: Path) -> dict:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read preprocessing catalog '{path.name}'.") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path.name} must contain a mapping.")
    return document


def load_preprocess_config(
    category_id: str,
    *,
    config_dir: Path | None = None,
    grayscale: bool | None = None,
    horizontal_flip_probability: float | None = None,
) -> PreprocessConfig:
    """Read size/statistics and category permissions, with explicit experiment overrides.

    Unknown category IDs or malformed settings fail. Overrides may enable the
    probability only when the category catalog explicitly allows horizontal flip.
    """
    directory = CONFIG_DIR if config_dir is None else Path(config_dir)
    defaults = _read_yaml(directory / "training_defaults.yaml")
    catalog = _read_yaml(directory / "categories.yaml")
    try:
        categories = catalog["categories"]
        if not isinstance(categories, list) or not all(
            isinstance(item, dict) and isinstance(item.get("id"), str) for item in categories
        ):
            raise ValueError("categories must be a list of catalog entries.")
        ids = [item["id"] for item in categories]
        if len(ids) != len(set(ids)):
            raise ValueError("Category IDs must be unique.")
        category = next((item for item in categories if item["id"] == category_id), None)
        if category is None:
            raise ValueError(f"Unknown category '{category_id}'.")
        settings = defaults["preprocessing"]
        normalization = defaults["normalization"]
        return PreprocessConfig(
            category_id=category_id,
            image_size=defaults["image_size"],
            mean=normalization["mean"],
            std=normalization["std"],
            grayscale=settings["grayscale"] if grayscale is None else grayscale,
            horizontal_flip_probability=(
                settings["horizontal_flip_probability"]
                if horizontal_flip_probability is None
                else horizontal_flip_probability
            ),
            allow_horizontal_flip=category["preprocessing"]["allow_horizontal_flip"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed preprocessing settings or category policy.") from error
