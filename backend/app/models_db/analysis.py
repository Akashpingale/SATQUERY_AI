"""The ``analyses`` table: metadata for each analysis request.

Rasters and large artifacts are NOT stored here - only metadata, status, the selected
model/tool, confidence, references to evidence/report files, and the execution trace.
Column types are portable across SQLite and PostgreSQL.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON as SA_JSON
from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..db.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Analysis(Base):
    """Persisted metadata for a single analysis."""

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    task: Mapped[str | None] = mapped_column(String(64), nullable=True)
    query: Mapped[str | None] = mapped_column(Text, nullable=True)

    model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_level: Mapped[str | None] = mapped_column(String(32), nullable=True)

    input_metadata: Mapped[dict[str, Any] | None] = mapped_column(SA_JSON, nullable=True)
    evidence_refs: Mapped[dict[str, Any] | None] = mapped_column(SA_JSON, nullable=True)
    trace: Mapped[dict[str, Any] | None] = mapped_column(SA_JSON, nullable=True)
    error: Mapped[dict[str, Any] | None] = mapped_column(SA_JSON, nullable=True)
    report_ref: Mapped[str | None] = mapped_column(String(256), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Analysis id={self.id} status={self.status} task={self.task}>"
