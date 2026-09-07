"""Shared API schemas, including the sanitized error envelope."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Sanitized error returned to clients (no stack traces, no secrets)."""

    error: ErrorDetail
