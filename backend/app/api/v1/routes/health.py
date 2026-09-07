"""Health endpoint: reports app info and database connectivity."""

from __future__ import annotations

from fastapi import APIRouter

from ....core.config import get_settings
from ....db.session import check_database_connection
from ....schemas.health import DatabaseStatus, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health() -> HealthResponse:
    settings = get_settings()
    connected, dialect = check_database_connection()
    return HealthResponse(
        status="ok" if connected else "degraded",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        device=settings.device,
        database=DatabaseStatus(connected=connected, dialect=dialect),
    )
