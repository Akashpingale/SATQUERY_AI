"""The ``raster_assets`` table: one row per uploaded raster.

Only metadata lives in the database. The raster bytes and the rendered preview stay
on the filesystem (or object storage later), referenced by relative paths.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON as SA_JSON
from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..db.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RasterAsset(Base):
    """An uploaded raster and its extracted metadata."""

    __tablename__ = "raster_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    original_filename: Mapped[str] = mapped_column(String(256))
    stored_filename: Mapped[str] = mapped_column(String(256))
    # Paths are relative to the configured storage root, never absolute.
    storage_path: Mapped[str] = mapped_column(String(512))
    preview_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    size_bytes: Mapped[int] = mapped_column(Integer)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # User-declared role/modality plus the metadata-derived hint (kept separate so a
    # guess is never presented as a user statement).
    role: Mapped[str] = mapped_column(String(32), default="single", index=True)
    modality: Mapped[str] = mapped_column(String(32), default="unknown")
    detected_modality: Mapped[str] = mapped_column(String(32), default="unknown")

    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    band_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dtype: Mapped[str | None] = mapped_column(String(32), nullable=True)
    driver: Mapped[str | None] = mapped_column(String(32), nullable=True)
    crs: Mapped[str | None] = mapped_column(String(128), nullable=True)
    crs_epsg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_georeferenced: Mapped[bool] = mapped_column(Boolean, default=False)

    # Full metadata payload (transform, bounds, resolution, bands, tags) and any
    # validation warnings raised at ingestion time.
    raster_metadata: Mapped[dict[str, Any] | None] = mapped_column(SA_JSON, nullable=True)
    warnings: Mapped[list[dict[str, Any]] | None] = mapped_column(SA_JSON, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<RasterAsset id={self.id} role={self.role} {self.width}x{self.height}>"
