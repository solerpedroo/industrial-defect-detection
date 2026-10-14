from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.api.routes import router
from app.core.config import get_settings
from app.core.upload_limit import InferenceUploadLimit
from app.inference import ServingRegistry, load_registry


def create_app() -> FastAPI:
    """Create the API; optional locally trusted serving installs at startup only."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.inference = await run_in_threadpool(
            load_registry,
            settings.inference_manifest,
            allow_synthetic=settings.allow_synthetic_inference,
        )
        yield
        application.state.inference = ServingRegistry()

    app = FastAPI(title="Industrial Defect Detection API", version="0.1.0", lifespan=lifespan)
    app.state.inference = ServingRegistry()
    app.add_middleware(InferenceUploadLimit)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    return app


app = create_app()
