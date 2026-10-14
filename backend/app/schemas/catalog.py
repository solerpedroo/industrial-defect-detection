from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    api_version: str


class DefectTypeCount(BaseModel):
    id: str
    test_images: int


class CategorySummary(BaseModel):
    id: str
    name: str
    kind: str
    purpose: str
    availability: Literal["missing", "present"]
    train_normal: int
    test_normal: int
    test_anomalous: int
    mask_count: int
    defect_types: list[DefectTypeCount]
    validation_split: Literal["not_created"]


class DatasetCatalogResponse(BaseModel):
    id: str
    name: str
    license: str
    license_url: str
    commercial_use: bool
    storage_root: str
    categories: list[CategorySummary]


class ModelCard(BaseModel):
    id: str
    name: str
    family: str
    status: Literal["planned", "ready"]
    backbone: str | None
    description: str


class ModelsResponse(BaseModel):
    models: list[ModelCard]


class ExperimentProtocol(BaseModel):
    id: str
    title: str
    status: Literal["planned", "complete"]
    question: str
    categories: list[str]
    models: list[str]
    metrics: list[str]


class ExperimentsResponse(BaseModel):
    experiments: list[ExperimentProtocol]
