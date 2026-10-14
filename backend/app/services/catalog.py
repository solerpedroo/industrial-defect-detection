"""Read YAML catalogs and local MVTec counts."""

from functools import lru_cache

import yaml

from app.core.paths import config_dir, dataset_root
from app.schemas.catalog import (
    CategorySummary,
    DatasetCatalogResponse,
    ExperimentProtocol,
    ModelCard,
)
from ml.datasets import summarize_category


def _read_yaml(name: str) -> dict:
    path = config_dir() / name
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{name} must contain a mapping.")
    return loaded


@lru_cache(maxsize=1)
def _category_file() -> dict:
    return _read_yaml("categories.yaml")


@lru_cache(maxsize=1)
def _model_file() -> dict:
    return _read_yaml("models.yaml")


@lru_cache(maxsize=1)
def _experiment_file() -> dict:
    return _read_yaml("experiments.yaml")


def dataset_catalog() -> DatasetCatalogResponse:
    """Return license metadata and per-category file counts."""
    document = _category_file()
    dataset = document["dataset"]
    categories = [_category_summary(item) for item in document["categories"]]
    return DatasetCatalogResponse(
        id=dataset["id"],
        name=dataset["name"],
        license=dataset["license"],
        license_url=dataset["license_url"],
        commercial_use=bool(dataset["commercial_use"]),
        storage_root=dataset["storage_root"],
        categories=categories,
    )


def category_summary(category_id: str) -> CategorySummary | None:
    for item in _category_file()["categories"]:
        if item["id"] == category_id:
            return _category_summary(item)
    return None


def models() -> list[ModelCard]:
    return [ModelCard.model_validate(item) for item in _model_file()["models"]]


def model_card(model_id: str) -> ModelCard | None:
    for model in models():
        if model.id == model_id:
            return model
    return None


def experiments() -> list[ExperimentProtocol]:
    return [ExperimentProtocol.model_validate(item) for item in _experiment_file()["experiments"]]


def experiment(experiment_id: str) -> ExperimentProtocol | None:
    for item in experiments():
        if item.id == experiment_id:
            return item
    return None


def _category_summary(item: dict) -> CategorySummary:
    counts = summarize_category(dataset_root(), item["id"])
    return CategorySummary(
        id=item["id"],
        name=item["name"],
        kind=item["kind"],
        purpose=item["purpose"],
        availability=counts["availability"],
        train_normal=counts["train_normal"],
        validation_normal=counts["validation_normal"],
        test_normal=counts["test_normal"],
        test_anomalous=counts["test_anomalous"],
        mask_count=counts["mask_count"],
        defect_types=counts["defect_types"],
        validation_split=counts["validation_split"],
    )
