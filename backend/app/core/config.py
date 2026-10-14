from dataclasses import dataclass
from os import getenv
from pathlib import Path

from app.core.paths import repo_root


@dataclass(frozen=True)
class Settings:
    """Runtime settings that are safe to read at application startup."""

    allowed_origins: tuple[str, ...]
    inference_manifest: Path | None = None
    allow_synthetic_inference: bool = False


def reports_root() -> Path:
    """Directory of generated report packages. Read on each request."""
    raw = getenv("REPORTS_ROOT")
    if raw:
        return Path(raw)
    return repo_root() / "sample_outputs"


def get_settings() -> Settings:
    raw_origins = getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    origins = tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
    manifest = getenv("INFERENCE_MANIFEST")
    flag = getenv("INFERENCE_ALLOW_SYNTHETIC", "0")
    if flag not in {"0", "1"}:
        raise ValueError("INFERENCE_ALLOW_SYNTHETIC must be 0 or 1.")
    return Settings(
        allowed_origins=origins,
        inference_manifest=Path(manifest) if manifest else None,
        allow_synthetic_inference=flag == "1",
    )
