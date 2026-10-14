"""Canonical autoencoder training configuration, without duplicated defaults."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from ml.preprocessing import PreprocessConfig, load_preprocess_config
from ml.preprocessing.config import CONFIG_DIR, PREPROCESS_VERSION


@dataclass(frozen=True)
class AutoencoderTrainConfig:
    preprocess: PreprocessConfig
    channels: tuple[int, int, int]
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    learning_rate: float
    seed: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.preprocess, PreprocessConfig)
            or self.preprocess.mean != (0, 0, 0)
            or self.preprocess.std != (1, 1, 1)
            or self.preprocess.image_size % 8
            or self.preprocess.image_size > 4096
        ):
            raise ValueError(
                "Autoencoder requires unit-interval RGB at a multiple of 8 up to 4096."
            )
        if (
            not isinstance(self.channels, (list, tuple))
            or len(self.channels) != 3
            or any(type(width) is not int or not 1 <= width <= 256 for width in self.channels)
        ):
            raise ValueError("channels must contain three integer widths in [1,256].")
        object.__setattr__(self, "channels", tuple(self.channels))
        for name in ("batch_size", "max_epochs", "early_stopping_patience"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            type(self.learning_rate) not in (int, float)
            or not 0 < self.learning_rate <= 1
            or not math.isfinite(self.learning_rate)
        ):
            raise ValueError("learning_rate must be finite and in (0,1].")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0,2**32).")

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "preprocess": self.preprocess.to_dict(),
            "channels": list(self.channels),
        }

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    @classmethod
    def from_dict(cls, document: dict) -> "AutoencoderTrainConfig":
        if not isinstance(document, dict) or set(document) != set(cls.__dataclass_fields__):
            raise ValueError("Invalid training configuration fields.")
        values = dict(document)
        preprocess = values["preprocess"]
        if not isinstance(preprocess, dict):
            raise ValueError("Invalid preprocessing configuration.")
        preprocess = dict(preprocess)
        if (
            type(preprocess.pop("schema_version", None)) is not int
            or document["preprocess"]["schema_version"] != PREPROCESS_VERSION
        ):
            raise ValueError("Unsupported preprocessing version.")
        try:
            values["preprocess"] = PreprocessConfig(**preprocess)
            return cls(**values)
        except (TypeError, KeyError) as error:
            raise ValueError("Malformed training configuration.") from error


def load_autoencoder_config(
    category_id: str, *, config_dir: Path | None = None
) -> AutoencoderTrainConfig:
    directory = CONFIG_DIR if config_dir is None else Path(config_dir)
    try:
        defaults = yaml.safe_load(
            (directory / "training_defaults.yaml").read_text(encoding="utf-8")
        )
        settings = defaults["autoencoder"]
        if settings["optimizer"] != "adam" or settings["loss"] != "mse":
            raise ValueError("Only Adam and MSE are implemented for the autoencoder.")
        preprocess = replace(
            load_preprocess_config(category_id, config_dir=directory),
            mean=settings["normalization"]["mean"],
            std=settings["normalization"]["std"],
        )
        return AutoencoderTrainConfig(
            preprocess=preprocess,
            channels=settings["channels"],
            batch_size=settings["batch_size"],
            max_epochs=settings["max_epochs"],
            early_stopping_patience=settings["early_stopping_patience"],
            learning_rate=settings["learning_rate"],
            seed=defaults["seed"],
        )
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, KeyError) as error:
        raise ValueError("Cannot load autoencoder training defaults.") from error
