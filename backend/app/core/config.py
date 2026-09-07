"""Application configuration.

All configuration comes from the environment (a local ``.env`` file in development,
real environment variables in Docker/production). Secrets are never hard-coded.
See ``.env.example`` for the documented set of variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: .../backend/app/core/config.py -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Environment-driven settings with safe local defaults.

    Defaults are chosen so the app is runnable with no ``.env`` at all: it falls
    back to a local SQLite database under ``storage/`` and CPU inference.

    Relative paths are anchored to the repository root rather than the current
    working directory, so the app behaves identically no matter where it is launched
    from (repo root, ``backend/``, or a container).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General
    app_env: str = "development"  # development | production | test
    app_name: str = "SatQuery AI"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    log_format: str = "console"  # json | console

    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = (
        "http://localhost:5173,http://localhost:4173,http://localhost:3000,"
        "http://localhost:8080"
    )

    # Database. Empty means "use the local SQLite default under storage/".
    # Docker/production set this to a postgresql+psycopg:// URL.
    database_url: str = ""

    # Storage / limits
    storage_dir: str = "storage"
    models_dir: str = "models"
    data_dir: str = "data"
    max_upload_mb: int = 200
    request_timeout_seconds: int = 120

    # Compute (Path A: CPU dev now, cloud GPU later)
    device: str = "cpu"  # cpu | cuda | auto

    # Change-VQA model selection. The deterministic measurement-grounded specialist is the
    # default. Options route the Change-VQA step to a learned model:
    #   deterministic  - measurement-grounded detector (default; preserves prior behaviour)
    #   cdvqa_transfer - Phase 15 improved classifier (CLIP + change_model BLIP features);
    #                    best learned model on the frozen CDVQA benchmark
    #   cdvqa          - Phase 9 trained classifier
    #   blip           - external generative BLIP change_model
    #   auto           - deterministic for measurement questions; the best learned model for
    #                    classification-style ("changed into what") questions
    change_vqa_model: str = "deterministic"

    @model_validator(mode="after")
    def _anchor_paths(self) -> Settings:
        """Resolve relative paths against the repo root and fill the DB default."""
        for field in ("storage_dir", "models_dir", "data_dir"):
            value = Path(getattr(self, field))
            if not value.is_absolute():
                value = PROJECT_ROOT / value
            object.__setattr__(self, field, str(value))

        if not self.database_url.strip():
            db_path = Path(self.storage_dir) / "satquery_dev.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            object.__setattr__(
                self, "database_url", f"sqlite:///{db_path.as_posix()}"
            )
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS origins parsed from the comma-separated env value."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def storage_path(self) -> Path:
        return Path(self.storage_dir)


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
