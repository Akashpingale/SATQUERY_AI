"""Health endpoint schemas."""

from __future__ import annotations

from pydantic import BaseModel


class DatabaseStatus(BaseModel):
    connected: bool
    dialect: str


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    app: str
    version: str
    environment: str
    device: str
    database: DatabaseStatus
