from typing import Literal

from pydantic import BaseModel, Field


class InferenceResponse(BaseModel):
    model_id: str
    category_id: str
    score: float = Field(ge=0, allow_inf_nan=False)
    label: Literal["normal", "anomalous"]
    threshold: float = Field(ge=0, allow_inf_nan=False)
    pixel_threshold: float = Field(ge=0, allow_inf_nan=False)
    input_png: str
    anomaly_map_png: str
    overlay_png: str
    mask_png: str
    reconstruction_png: str | None
    display_max: float = Field(ge=0, allow_inf_nan=False)
    map_width: int
    map_height: int
    data_kind: Literal["synthetic", "mvtec_ad"]
    artifact_sha256: str
    calibration_sha256: str
    config_hash: str
    inference_ms: float = Field(ge=0, allow_inf_nan=False)
    request_ms: float = Field(ge=0, allow_inf_nan=False)


class CompareResponse(BaseModel):
    category_id: str
    results: list[InferenceResponse]
    request_ms: float = Field(ge=0, allow_inf_nan=False)
