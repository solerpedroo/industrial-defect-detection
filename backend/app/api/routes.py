import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import reports_root
from app.core.images import MAX_UPLOAD_BYTES, DecodedImage, ImageValidationError, decode_upload
from app.inference import ServingRegistry
from app.schemas.catalog import (
    CategorySummary,
    DatasetCatalogResponse,
    ExperimentProtocol,
    ExperimentsResponse,
    HealthResponse,
    ModelsResponse,
    ReportResponse,
)
from app.schemas.inference import CompareResponse, InferenceResponse
from app.services.catalog import (
    category_summary,
    dataset_catalog,
    experiment,
    experiments,
    has_category,
    model_card,
    models,
)
from app.services.reports import ReportNotFoundError, ReportTooLargeError, load_report
from app.visualization import render_prediction

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
def list_models(request: Request) -> ModelsResponse:
    """Return catalog models and whether an artifact can be served."""
    registry = request.app.state.inference
    cards = []
    for model in models():
        ready = registry.categories(model.id)
        cards.append(
            model.model_copy(
                update={
                    "status": "ready" if ready else "planned",
                    "ready_categories": ready,
                    "synthetic_categories": registry.categories(model.id, synthetic=True),
                }
            )
        )
    return ModelsResponse(models=cards)


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


@router.get("/reports/{experiment_id}", response_model=ReportResponse, tags=["experiments"])
def report(experiment_id: str) -> ReportResponse:
    """Return a generated Markdown report. Does not train, score, or read the dataset."""
    if experiment(experiment_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment '{experiment_id}'.")
    try:
        payload = load_report(experiment_id, reports_root())
    except ReportNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"No report artifact has been generated for '{experiment_id}'.",
        ) from error
    except ReportTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    return ReportResponse.model_validate(payload)


@router.post("/inference", response_model=InferenceResponse, tags=["inference"])
async def inference(
    request: Request,
    image: UploadFile = File(...),
    category_id: str = Form(...),
    model_id: str = Form(...),
) -> InferenceResponse:
    """Serve one cached calibrated predictor; never fit, reload or recalibrate."""
    started = time.perf_counter()
    raw_bytes = await _read_upload(image)
    decoded = await run_in_threadpool(_validate_inference_request, raw_bytes, category_id, model_id)
    registry = request.app.state.inference
    _require_installed(registry, category_id, model_id)
    try:
        result = await run_in_threadpool(_serve, registry, decoded, category_id, model_id)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=503, detail="Inference could not produce a valid result."
        ) from error
    return InferenceResponse(**result, request_ms=(time.perf_counter() - started) * 1000)


@router.post("/inference/compare", response_model=CompareResponse, tags=["inference"])
async def compare_inference(
    request: Request,
    image: UploadFile = File(...),
    category_id: str = Form(...),
    model_a: str = Form(...),
    model_b: str = Form(...),
) -> CompareResponse:
    """Reuse installed predictors for one ephemeral upload, not saved experiments."""
    started = time.perf_counter()
    raw_bytes = await _read_upload(image)
    if model_a == model_b:
        raise HTTPException(status_code=422, detail="Choose two different models to compare.")
    decoded = await run_in_threadpool(_validate_inference_request, raw_bytes, category_id, model_a)
    _require_model(model_b)
    registry = request.app.state.inference
    for model_id in (model_a, model_b):
        _require_installed(registry, category_id, model_id)
    try:
        results = await run_in_threadpool(
            lambda: [_serve(registry, decoded, category_id, model) for model in (model_a, model_b)]
        )
    except (ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=503, detail="Inference could not produce valid results."
        ) from error
    elapsed = (time.perf_counter() - started) * 1000
    return CompareResponse(
        category_id=category_id,
        request_ms=elapsed,
        results=[InferenceResponse(**result, request_ms=elapsed) for result in results],
    )


async def _read_upload(image: UploadFile) -> bytes:
    try:
        return await image.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await image.close()


def _validate_inference_request(raw_bytes: bytes, category_id: str, model_id: str) -> DecodedImage:
    if not has_category(category_id):
        raise HTTPException(status_code=422, detail=f"Unknown category '{category_id}'.")
    _require_model(model_id)
    try:
        return decode_upload(raw_bytes)
    except ImageValidationError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


def _require_model(model_id: str) -> None:
    if model_card(model_id) is None:
        raise HTTPException(status_code=422, detail=f"Unknown model '{model_id}'.")


def _require_installed(registry: ServingRegistry, category: str, model: str):
    try:
        registry.require(category, model)
    except LookupError as error:
        raise HTTPException(status_code=501, detail=NOT_READY) from error


def _serve(registry: ServingRegistry, decoded: DecodedImage, category: str, model: str) -> dict:
    prediction, installed, elapsed = registry.predict(category, model, decoded.image)
    rendered = render_prediction(
        prediction, size=installed.size, pixel_threshold=installed.pixel_threshold
    )
    return {
        "model_id": model,
        "category_id": category,
        "score": float(prediction.score),
        "label": "anomalous" if prediction.score > installed.image_threshold else "normal",
        "threshold": installed.image_threshold,
        "pixel_threshold": installed.pixel_threshold,
        "map_width": installed.size,
        "map_height": installed.size,
        "data_kind": installed.data_kind,
        "artifact_sha256": installed.artifact_sha256,
        "calibration_sha256": installed.calibration_sha256,
        "config_hash": installed.config_hash,
        "inference_ms": elapsed,
        **rendered,
    }
