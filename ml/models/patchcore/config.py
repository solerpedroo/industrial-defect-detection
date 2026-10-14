"""Validated YAML-led configuration for the bounded ResNet-18 PatchCore adapter."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from ml.preprocessing import PreprocessConfig, load_preprocess_config
from ml.preprocessing.config import CONFIG_DIR, PREPROCESS_VERSION

LAYERS = ("layer1", "layer2", "layer3", "layer4")


@dataclass(frozen=True)
class PatchcoreConfig:
    preprocess: PreprocessConfig
    backbone: str
    layers: tuple[str, ...]
    pre_trained: bool
    coreset_sampling_ratio: float
    num_neighbors: int
    batch_size: int
    max_patches: int
    seed: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.preprocess, PreprocessConfig)
            or not 32 <= self.preprocess.image_size <= 4096
            or self.preprocess.image_size % 32
            or self.preprocess.horizontal_flip_probability != 0
        ):
            raise ValueError("PatchCore needs unaugmented RGB at a multiple of 32 in [32,4096].")
        if self.backbone != "resnet18":
            raise ValueError("Only the planned ResNet-18 PatchCore backbone is implemented.")
        if (
            not isinstance(self.layers, (list, tuple))
            or not self.layers
            or any(layer not in LAYERS for layer in self.layers)
            or tuple(sorted(set(self.layers), key=LAYERS.index)) != tuple(self.layers)
        ):
            raise ValueError("Feature layers must be unique ResNet stages in increasing order.")
        object.__setattr__(self, "layers", tuple(self.layers))
        if type(self.pre_trained) is not bool:
            raise ValueError("pre_trained must be boolean; false is explicit regression mode.")
        ratio = self.coreset_sampling_ratio
        if type(ratio) not in (int, float) or not 0 < ratio <= 1:
            raise ValueError("Coreset ratio must be finite and in (0,1].")
        object.__setattr__(self, "coreset_sampling_ratio", float(ratio))
        for name in ("num_neighbors", "batch_size", "max_patches"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0,2**32).")

    @property
    def patches_per_image(self) -> int:
        stride = 2 ** (LAYERS.index(self.layers[0]) + 2)
        return (self.preprocess.image_size // stride) ** 2

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "preprocess": self.preprocess.to_dict(),
            "layers": list(self.layers),
        }

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    @classmethod
    def from_dict(cls, document: dict) -> "PatchcoreConfig":
        if not isinstance(document, dict) or set(document) != set(cls.__dataclass_fields__):
            raise ValueError("Invalid PatchCore configuration fields.")
        values = dict(document)
        preprocessing = values["preprocess"]
        if not isinstance(preprocessing, dict):
            raise ValueError("Invalid preprocessing configuration.")
        preprocessing = dict(preprocessing)
        version = preprocessing.pop("schema_version", None)
        if type(version) is not int or version != PREPROCESS_VERSION:
            raise ValueError("Unsupported preprocessing version.")
        try:
            values["preprocess"] = PreprocessConfig(**preprocessing)
            return cls(**values)
        except (KeyError, TypeError) as error:
            raise ValueError("Malformed PatchCore configuration.") from error


def load_patchcore_config(category_id: str, *, config_dir: Path | None = None) -> PatchcoreConfig:
    directory = CONFIG_DIR if config_dir is None else Path(config_dir)
    try:
        document = yaml.safe_load(
            (directory / "training_defaults.yaml").read_text(encoding="utf-8")
        )
        settings = document["patchcore"]
        return PatchcoreConfig(
            preprocess=load_preprocess_config(category_id, config_dir=directory),
            backbone=settings["backbone"],
            layers=settings["layers"],
            pre_trained=settings["pre_trained"],
            coreset_sampling_ratio=settings["coreset_sampling_ratio"],
            num_neighbors=settings["num_neighbors"],
            batch_size=settings["batch_size"],
            max_patches=settings["max_patches"],
            seed=document["seed"],
        )
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError) as error:
        raise ValueError("Cannot load PatchCore defaults or category configuration.") from error
