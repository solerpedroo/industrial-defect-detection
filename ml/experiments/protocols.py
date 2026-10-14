"""Load the recorded experiment B and C protocol files."""

from pathlib import Path

import yaml

from ml.experiments import PROTOCOL_IDS

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str, protocol_id: str) -> dict:
    path = REPO_ROOT / relative
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("protocol_id") != protocol_id:
        raise ValueError(f"{relative} is not protocol {protocol_id}.")
    if protocol_id not in PROTOCOL_IDS:
        raise ValueError("Unknown protocol id.")
    return document


def load_robustness_config() -> dict:
    """Return the documented noise, blur and brightness grid."""
    document = _load("experiments/robustness/config.yaml", "exp-b-robustness")
    grid = document.get("degradations")
    expected = {
        "gaussian_noise_sigma": [10, 20, 30],
        "gaussian_blur_sigma": [1, 2, 3],
        "brightness_factors": [0.6, 0.8, 1.2],
    }
    if grid != expected:
        raise ValueError("Robustness intensities no longer match the recorded protocol.")
    return document


def degradation_conditions() -> tuple[tuple[str, float | None], ...]:
    """Original reference plus every recorded intensity, in protocol order."""
    document = load_robustness_config()
    grid = document["degradations"]
    conditions: list[tuple[str, float | None]] = [("original", None)]
    conditions.extend(("gaussian_noise", float(value)) for value in grid["gaussian_noise_sigma"])
    conditions.extend(("gaussian_blur", float(value)) for value in grid["gaussian_blur_sigma"])
    conditions.extend(("brightness", float(value)) for value in grid["brightness_factors"])
    return tuple(conditions)


def load_dataset_size_config() -> dict:
    """Return the recorded normal-training fractions and repeat count."""
    document = _load("experiments/dataset-size/config.yaml", "exp-c-dataset-size")
    if document.get("normal_train_fractions") != [0.25, 0.5, 0.75, 1.0] or document.get("repeats") != 3:
        raise ValueError("Dataset-size fractions or repeats no longer match the recorded protocol.")
    if document.get("seed") != 42:
        raise ValueError("Dataset-size seed no longer matches the recorded protocol.")
    return document
