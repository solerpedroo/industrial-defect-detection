"""Protocol-led experiment A configuration, separate from model defaults."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2] / "experiments/autoencoder-vs-patchcore/config.yaml"
)


@dataclass(frozen=True)
class EvaluationConfig:
    protocol_id: str
    seed: int
    categories: tuple[str, ...]
    models: tuple[str, ...]
    image_size: int
    target_image_fpr: float
    target_pixel_fpr: float
    aupro_fpr_limit: float
    warmup_images: int
    max_pixels: int

    def __post_init__(self):
        if self.protocol_id != "exp-a-autoencoder-vs-patchcore":
            raise ValueError("Only experiment A is implemented.")
        catalog = yaml.safe_load(
            (PROTOCOL_PATH.parents[2] / "ml/configs/categories.yaml").read_text()
        )
        known = {item["id"] for item in catalog["categories"]}
        if (
            not isinstance(self.categories, (list, tuple))
            or not self.categories
            or any(not isinstance(category, str) for category in self.categories)
            or len(set(self.categories)) != len(self.categories)
            or not set(self.categories) <= known
        ):
            raise ValueError("Protocol categories must be unique catalog IDs.")
        if not isinstance(self.models, (list, tuple)) or tuple(self.models) != (
            "autoencoder",
            "patchcore",
        ):
            raise ValueError("Experiment A compares Autoencoder then PatchCore.")
        object.__setattr__(self, "categories", tuple(self.categories))
        object.__setattr__(self, "models", tuple(self.models))
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be in [0,2**32).")
        if (
            type(self.image_size) is not int
            or not 32 <= self.image_size <= 4096
            or self.image_size % 32
        ):
            raise ValueError("Evaluation resolution must be a multiple of 32 in [32,4096].")
        for value in (self.target_image_fpr, self.target_pixel_fpr):
            if type(value) not in (int, float) or not 0 < value < 1:
                raise ValueError("Target false-positive rates must be in (0,1).")
        if type(self.aupro_fpr_limit) not in (int, float) or not 0 < self.aupro_fpr_limit <= 1:
            raise ValueError("AUPRO FPR limit must be in (0,1].")
        for value in (self.warmup_images, self.max_pixels):
            if type(value) is not int or value <= 0:
                raise ValueError("Warmup and pixel budget must be positive integers.")

    def to_dict(self):
        return asdict(self)

    @property
    def config_hash(self):
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()


def load_evaluation_config(path: Path = PROTOCOL_PATH) -> EvaluationConfig:
    try:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if (
            values.pop("split_policy")
            != "validation thresholds, then a single official test report"
        ):
            raise ValueError("Unsupported split policy.")
        return EvaluationConfig(**values)
    except (OSError, yaml.YAMLError, TypeError, KeyError, AttributeError) as error:
        raise ValueError("Invalid evaluation protocol configuration.") from error
