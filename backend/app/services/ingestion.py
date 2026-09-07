"""Raster ingestion: validate, extract metadata, render a preview, persist.

Order matters for safety: the file is streamed to disk under a sanitized name, then
validated as a real raster. If validation fails the file is deleted and nothing is
persisted, so invalid uploads leave no trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from geospatial import (
    RasterMetadata,
    ValidationResult,
    infer_modality,
    render_preview,
    validate_raster_file,
)
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..models_db.raster_asset import RasterAsset
from ..security.files import build_storage_name
from . import storage

logger = get_logger("satquery.ingestion")

VALID_ROLES = {"single", "optical", "sar", "t1", "t2"}
VALID_MODALITIES = {"optical", "sar", "unknown"}


@dataclass
class IngestionResult:
    """Outcome of an ingestion attempt."""

    ok: bool
    asset: RasterAsset | None = None
    validation: ValidationResult | None = None


def _preview_relative_path(asset_id: str) -> str:
    return f"{storage.EVIDENCE_DIR}/previews/{asset_id}.png"


def ingest_upload(
    db: Session,
    source: BinaryIO,
    original_filename: str,
    content_type: str | None = None,
    role: str = "single",
    modality: str = "unknown",
) -> IngestionResult:
    """Ingest one uploaded raster.

    Returns an :class:`IngestionResult`. On failure the stored bytes are removed and
    no database row is created; the validation issues explain why in plain language.
    """
    asset = RasterAsset()
    asset_id = asset.id or None
    storage_name = build_storage_name(original_filename, asset_id)

    stored = storage.save_upload_stream(source, storage_name, subdir=storage.UPLOADS_DIR)

    validation = validate_raster_file(stored.absolute_path, original_filename=original_filename)
    if not validation.valid or validation.metadata is None:
        storage.delete_stored(stored.relative_path)
        logger.info(
            "Rejected upload '%s': %s",
            original_filename,
            ",".join(issue.code for issue in validation.errors),
        )
        return IngestionResult(ok=False, validation=validation)

    metadata: RasterMetadata = validation.metadata

    # Preview rendering is best-effort: a preview failure must not reject a valid raster.
    preview_relative: str | None = _preview_relative_path(asset.id)
    try:
        preview_absolute = storage.absolute_path(preview_relative)
        render_preview(stored.absolute_path, preview_absolute)
    except Exception:  # noqa: BLE001 - preview is a convenience, not a requirement
        logger.warning("Preview rendering failed for asset %s", asset.id, exc_info=True)
        preview_relative = None
        validation.add(
            "preview_unavailable",
            "A visual preview could not be generated for this file.",
            severity="warning",
        )

    detected = infer_modality(metadata)
    requested_modality = modality if modality in VALID_MODALITIES else "unknown"

    asset.original_filename = original_filename[:256]
    asset.stored_filename = storage_name
    asset.storage_path = stored.relative_path
    asset.preview_path = preview_relative
    asset.size_bytes = stored.size_bytes
    asset.content_type = content_type
    asset.role = role if role in VALID_ROLES else "single"
    # Respect an explicit user declaration; otherwise fall back to the detected hint.
    asset.modality = requested_modality if requested_modality != "unknown" else str(detected)
    asset.detected_modality = str(detected)
    asset.width = metadata.width
    asset.height = metadata.height
    asset.band_count = metadata.band_count
    asset.dtype = metadata.dtype
    asset.driver = metadata.driver
    asset.crs = metadata.crs
    asset.crs_epsg = metadata.crs_epsg
    asset.is_georeferenced = metadata.is_georeferenced
    asset.raster_metadata = metadata.to_dict()
    asset.warnings = [
        {"code": i.code, "message": i.message, "severity": i.severity} for i in validation.warnings
    ]

    db.add(asset)
    db.commit()
    db.refresh(asset)

    logger.info(
        "Ingested asset %s (%sx%s, %s bands, georeferenced=%s)",
        asset.id,
        asset.width,
        asset.height,
        asset.band_count,
        asset.is_georeferenced,
    )
    return IngestionResult(ok=True, asset=asset, validation=validation)


def metadata_from_asset(asset: RasterAsset) -> RasterMetadata | None:
    """Rebuild a :class:`RasterMetadata` from a stored asset row.

    Falls back to re-reading the file when the stored payload is missing.
    """
    payload = asset.raster_metadata
    if payload:
        bands = payload.get("bands") or []
        from geospatial.types import BandInfo  # local import to avoid cycles at module load

        return RasterMetadata(
            width=payload["width"],
            height=payload["height"],
            band_count=payload["band_count"],
            dtype=payload["dtype"],
            driver=payload["driver"],
            is_georeferenced=payload["is_georeferenced"],
            crs=payload.get("crs"),
            crs_epsg=payload.get("crs_epsg"),
            transform=tuple(payload["transform"]) if payload.get("transform") else None,
            bounds=tuple(payload["bounds"]) if payload.get("bounds") else None,
            resolution=tuple(payload["resolution"]) if payload.get("resolution") else None,
            nodata=payload.get("nodata"),
            bands=[BandInfo(**band) for band in bands],
            tags=payload.get("tags") or {},
        )

    try:
        path = storage.absolute_path(asset.storage_path)
    except ValueError:
        return None
    if not Path(path).is_file():
        return None
    from geospatial import read_metadata

    return read_metadata(path)
