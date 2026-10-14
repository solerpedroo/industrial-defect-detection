from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.images import ImageValidationError, decode_upload
from app.schemas.catalog import (
    CategorySummary,
    DatasetCatalogResponse,
    ExperimentProtocol,
    ExperimentsResponse,
    HealthResponse,
    ModelsResponse,
)
from app.services.catalog import (
    category_summary,
    dataset_catalog,
    experiment,
    experiments,
    model_card,
    models,
)

router = APIRouter(prefix="/api")
API_VERSION = "0.1.0"
NOT_READY = "No trained artifact is available for this model yet."


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Return liveness data without exposing environment configuration."""
    return HealthResponse(status="ok", api_version=API_VERSION)


@router.get("/datasets", response_model=DatasetCatalogResponse, tags=["datasets"])
def list_datasets() -> DatasetCatalogResponse:
    """Return the MVTec catalog and local file counts."""
    return dataset_catalog()


@router.get("/datasets/{category_id}/summary", response_model=CategorySummary, tags=["datasets"])
def dataset_summary(category_id: str) -> CategorySummary:
    """Return one category summary."""
    summary = category_summary(category_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Unknown category '{category_id}'.")
    return summary


@router.get("/models", response_model=ModelsResponse, tags=["models"])
def list_models() -> ModelsResponse:
    """Return catalog models and whether an artifact can be served."""
    return ModelsResponse(models=models())


@router.get("/experiments", response_model=ExperimentsResponse, tags=["experiments"])
def list_experiments() -> ExperimentsResponse:
    """Return scientific protocols. Measured metrics are not included."""
    return ExperimentsResponse(experiments=experiments())


@router.get("/experiments/{experiment_id}", response_model=ExperimentProtocol, tags=["experiments"])
def experiment_detail(experiment_id: str) -> ExperimentProtocol:
    """Return one protocol."""
    protocol = experiment(experiment_id)
    if protocol is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment '{experiment_id}'.")
    return protocol


@router.get("/reports/{experiment_id}", tags=["experiments"])
def report(experiment_id: str) -> None:
    """Reports are generated in Wave 8."""
    if experiment(experiment_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment '{experiment_id}'.")
    raise HTTPException(
        status_code=404,
        detail=f"No report artifact has been generated for '{experiment_id}'.",
    )


@router.post("/inference", tags=["inference"])
async def inference(
    image: UploadFile = File(...),
    category_id: str = Form(...),
    model_id: str = Form(...),
) -> None:
    """Validate an inspection request. Scoring waits for a ready artifact."""
    _validate_inference_request(await image.read(), category_id, model_id)
    raise HTTPException(status_code=501, detail=NOT_READY)


@router.post("/inference/compare", tags=["inference"])
async def compare_inference(
    image: UploadFile = File(...),
    category_id: str = Form(...),
    model_a: str = Form(...),
    model_b: str = Form(...),
) -> None:
    """Validate a two-model request. Scoring waits for ready artifacts."""
    if model_a == model_b:
        raise HTTPException(status_code=422, detail="Choose two different models to compare.")
    raw_bytes = await image.read()
    _validate_inference_request(raw_bytes, category_id, model_a)
    _require_model(model_b)
    raise HTTPException(status_code=501, detail=NOT_READY)


def _validate_inference_request(raw_bytes: bytes, category_id: str, model_id: str) -> None:
    if category_summary(category_id) is None:
        raise HTTPException(status_code=422, detail=f"Unknown category '{category_id}'.")
    _require_model(model_id)
    try:
        decode_upload(raw_bytes)
    except ImageValidationError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


def _require_model(model_id: str) -> None:
    if model_card(model_id) is None:
        raise HTTPException(status_code=422, detail=f"Unknown model '{model_id}'.")
