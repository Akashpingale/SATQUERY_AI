"""Raster upload, metadata, preview, and pair-compatibility endpoints.

NOTE (security): these endpoints are unauthenticated. That is acceptable for local
development and the evaluation demo, but authentication/authorization must be added
before any public deployment. Uploads are validated, size-limited, and stored under
sanitized names confined to the storage root.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from geospatial import check_pair_compatibility
from sqlalchemy import select
from sqlalchemy.orm import Session

from ....core.config import get_settings
from ....db.session import get_db
from ....models_db.raster_asset import RasterAsset
from ....schemas.uploads import (
    BandOut,
    CompatibilityRequest,
    CompatibilityResponse,
    GeoreferenceOut,
    IssueOut,
    RasterAssetOut,
    UploadListResponse,
    UploadResponse,
)
from ....security.files import UnsafePathError, is_allowed_content_type
from ....services import storage
from ....services.ingestion import (
    VALID_MODALITIES,
    VALID_ROLES,
    ingest_upload,
    metadata_from_asset,
)

router = APIRouter(tags=["uploads"])


def _to_out(asset: RasterAsset, api_prefix: str) -> RasterAssetOut:
    """Map an ORM asset onto the API representation."""
    payload = asset.raster_metadata or {}
    return RasterAssetOut(
        id=asset.id,
        created_at=asset.created_at,
        original_filename=asset.original_filename,
        size_bytes=asset.size_bytes,
        role=asset.role,  # type: ignore[arg-type]
        modality=asset.modality,  # type: ignore[arg-type]
        detected_modality=asset.detected_modality,  # type: ignore[arg-type]
        width=asset.width,
        height=asset.height,
        band_count=asset.band_count,
        dtype=asset.dtype,
        driver=asset.driver,
        georeference=GeoreferenceOut(
            is_georeferenced=asset.is_georeferenced,
            crs=asset.crs,
            crs_epsg=asset.crs_epsg,
            transform=list(payload["transform"]) if payload.get("transform") else None,
            bounds=list(payload["bounds"]) if payload.get("bounds") else None,
            resolution=list(payload["resolution"]) if payload.get("resolution") else None,
        ),
        bands=[BandOut(**band) for band in (payload.get("bands") or [])],
        nodata=payload.get("nodata"),
        tags=payload.get("tags") or {},
        preview_url=(f"{api_prefix}/uploads/{asset.id}/preview" if asset.preview_path else None),
        warnings=[IssueOut(**w) for w in (asset.warnings or [])],
    )


def _get_asset_or_404(db: Session, asset_id: str) -> RasterAsset:
    asset = db.get(RasterAsset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Upload not found.")
    return asset


@router.post(
    "/uploads",
    response_model=UploadResponse,
    status_code=201,
    summary="Upload a raster (GeoTIFF/TIFF, or PNG/JPEG for benchmark data)",
)
async def create_upload(
    file: UploadFile = File(..., description="Raster file to ingest"),
    role: str = Form("single", description="single | optical | sar | t1 | t2"),
    modality: str = Form("unknown", description="optical | sar | unknown"),
    db: Session = Depends(get_db),
) -> UploadResponse:
    settings = get_settings()

    if role not in VALID_ROLES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid role '{role}'. Expected one of: {', '.join(sorted(VALID_ROLES))}.",
        )
    if modality not in VALID_MODALITIES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid modality '{modality}'. Expected one of: "
                f"{', '.join(sorted(VALID_MODALITIES))}."
            ),
        )
    if not is_allowed_content_type(file.content_type):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Content type '{file.content_type}' is not accepted. Upload a GeoTIFF/TIFF, "
                "PNG, or JPEG file."
            ),
        )

    try:
        result = ingest_upload(
            db=db,
            source=file.file,
            original_filename=file.filename or "upload",
            content_type=file.content_type,
            role=role,
            modality=modality,
        )
    except storage.UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail="Invalid file name.") from exc
    finally:
        await file.close()

    if not result.ok or result.asset is None:
        issues = result.validation.errors if result.validation else []
        raise HTTPException(
            status_code=422,
            detail={
                "message": "The file could not be used as a remote-sensing raster.",
                "issues": [
                    {"code": i.code, "message": i.message, "severity": i.severity} for i in issues
                ],
            },
        )

    return UploadResponse(asset=_to_out(result.asset, settings.api_v1_prefix))


@router.get("/uploads", response_model=UploadListResponse, summary="List uploaded rasters")
def list_uploads(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> UploadListResponse:
    prefix = get_settings().api_v1_prefix
    stmt = select(RasterAsset).order_by(RasterAsset.created_at.desc()).limit(limit)
    assets = list(db.execute(stmt).scalars())
    return UploadListResponse(
        count=len(assets), assets=[_to_out(a, prefix) for a in assets]
    )


@router.get(
    "/uploads/{asset_id}",
    response_model=RasterAssetOut,
    summary="Get an upload's metadata",
)
def get_upload(asset_id: str, db: Session = Depends(get_db)) -> RasterAssetOut:
    asset = _get_asset_or_404(db, asset_id)
    return _to_out(asset, get_settings().api_v1_prefix)


@router.get(
    "/uploads/{asset_id}/preview",
    summary="Get the rendered PNG preview of an upload",
    response_class=FileResponse,
)
def get_preview(asset_id: str, db: Session = Depends(get_db)) -> FileResponse:
    asset = _get_asset_or_404(db, asset_id)
    if not asset.preview_path:
        raise HTTPException(status_code=404, detail="No preview is available for this upload.")
    try:
        path = storage.absolute_path(asset.preview_path)
    except UnsafePathError as exc:
        raise HTTPException(status_code=404, detail="Preview not available.") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Preview file is missing.")
    return FileResponse(path, media_type="image/png", filename=f"{asset_id}.png")


@router.delete("/uploads/{asset_id}", status_code=204, summary="Delete an upload")
def delete_upload(asset_id: str, db: Session = Depends(get_db)) -> None:
    asset = _get_asset_or_404(db, asset_id)
    storage.delete_stored(asset.storage_path)
    if asset.preview_path:
        storage.delete_stored(asset.preview_path)
    db.delete(asset)
    db.commit()


@router.post(
    "/uploads/compatibility",
    response_model=CompatibilityResponse,
    summary="Check whether two uploads can be analyzed together",
)
def check_compatibility(
    request: CompatibilityRequest,
    db: Session = Depends(get_db),
) -> CompatibilityResponse:
    """Validate a temporal or optical-SAR pair before any joint analysis.

    This is the guard that prevents change/fusion analysis from silently running on
    images that do not describe the same area.
    """
    if request.asset_id_a == request.asset_id_b:
        raise HTTPException(status_code=422, detail="Provide two different uploads.")

    asset_a = _get_asset_or_404(db, request.asset_id_a)
    asset_b = _get_asset_or_404(db, request.asset_id_b)

    meta_a = metadata_from_asset(asset_a)
    meta_b = metadata_from_asset(asset_b)
    if meta_a is None or meta_b is None:
        raise HTTPException(
            status_code=409,
            detail="Metadata is unavailable for one of the uploads; re-upload the file.",
        )

    label_a = "optical image" if request.purpose == "optical_sar" else "first image (T1)"
    label_b = "SAR image" if request.purpose == "optical_sar" else "second image (T2)"
    result = check_pair_compatibility(meta_a, meta_b, label_a=label_a, label_b=label_b)

    issues = [
        IssueOut(code=i.code, message=i.message, severity=i.severity) for i in result.issues
    ]

    # For optical-SAR fusion both modalities must actually be present.
    if request.purpose == "optical_sar":
        modalities = {asset_a.modality, asset_b.modality}
        if not {"optical", "sar"} <= modalities:
            issues.append(
                IssueOut(
                    code="modality_mismatch",
                    message=(
                        "Optical-SAR analysis needs one optical image and one SAR image. "
                        f"Received: {', '.join(sorted(modalities))}."
                    ),
                    severity="error",
                )
            )

    compatible = result.compatible and not any(i.severity == "error" for i in issues)

    def summary(asset: RasterAsset) -> dict[str, object]:
        return {
            "id": asset.id,
            "filename": asset.original_filename,
            "modality": asset.modality,
            "width": asset.width,
            "height": asset.height,
            "crs": asset.crs,
            "is_georeferenced": asset.is_georeferenced,
        }

    return CompatibilityResponse(
        compatible=compatible,
        purpose=request.purpose,
        overlap_ratio=result.overlap_ratio,
        same_crs=result.same_crs,
        resampling_required=result.resampling_required,
        alignment_required=result.alignment_required,
        issues=issues,
        notes=result.notes,
        asset_a=summary(asset_a),
        asset_b=summary(asset_b),
    )
