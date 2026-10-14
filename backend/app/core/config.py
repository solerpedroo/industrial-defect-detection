from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings that are safe to read at application startup."""

    allowed_origins: tuple[str, ...]


def get_settings() -> Settings:
    raw_origins = getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    origins = tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
    return Settings(allowed_origins=origins)
