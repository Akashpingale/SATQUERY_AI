"""Schemas for raster upload, metadata, and pair-compatibility endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

RoleLiteral = Literal["single", "optical", "sar", "t1", "t2"]
ModalityLiteral = Literal["optical", "sar", "unknown"]


class IssueOut(BaseModel):
    code: str
    message: str
    severity: str


class BandOut(BaseModel):
    index: int
    dtype: str
    nodata: float | None = None
    description: str | None = None


class GeoreferenceOut(BaseModel):
    """Georeferencing block. All fields are None for non-georeferenced images."""

    is_georeferenced: bool
    crs: str | None = None
    crs_epsg: int | None = None
    transform: list[float] | None = None
    bounds: list[float] | None = Field(default=None, description="left, bottom, right, top")
    resolution: list[float] | None = None


class RasterAssetOut(BaseModel):
    id: str
    created_at: datetime
    original_filename: str
    size_bytes: int
    role: RoleLiteral
    modality: ModalityLiteral
    detected_modality: ModalityLiteral
    width: int | None
    height: int | None
    band_count: int | None
    dtype: str | None
    driver: str | None
    georeference: GeoreferenceOut
    bands: list[BandOut] = Field(default_factory=list)
    nodata: float | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    preview_url: str | None = None
    warnings: list[IssueOut] = Field(default_factory=list)


class UploadResponse(BaseModel):
    asset: RasterAssetOut


class UploadListResponse(BaseModel):
    count: int
    assets: list[RasterAssetOut]


class CompatibilityRequest(BaseModel):
    asset_id_a: str
    asset_id_b: str
    purpose: Literal["temporal", "optical_sar"] = "temporal"


class CompatibilityResponse(BaseModel):
    compatible: bool
    purpose: str
    overlap_ratio: float | None = None
    same_crs: bool | None = None
    resampling_required: bool
    alignment_required: bool
    issues: list[IssueOut] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    asset_a: dict[str, Any] = Field(default_factory=dict)
    asset_b: dict[str, Any] = Field(default_factory=dict)
