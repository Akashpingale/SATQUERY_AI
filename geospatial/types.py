"""Shared geospatial data types.

Plain dataclasses so this package stays independent of the web layer; the backend
maps these onto Pydantic schemas at the API edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Modality(StrEnum):
    """Imagery modality. Mirrors the backend task registry values."""

    OPTICAL = "optical"
    SAR = "sar"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BandInfo:
    """Per-band description."""

    index: int  # 1-based, as in GDAL/Rasterio
    dtype: str
    nodata: float | None = None
    description: str | None = None


@dataclass(frozen=True)
class RasterMetadata:
    """Metadata describing a raster, independent of any storage concern.

    ``crs`` / ``transform`` / ``bounds`` / ``resolution`` are ``None`` for plain
    (non-georeferenced) images such as benchmark PNG/JPEG inputs. Consumers must
    treat missing georeferencing explicitly rather than assuming defaults.
    """

    width: int
    height: int
    band_count: int
    dtype: str
    driver: str
    is_georeferenced: bool
    crs: str | None = None
    crs_epsg: int | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    bounds: tuple[float, float, float, float] | None = None  # left, bottom, right, top
    resolution: tuple[float, float] | None = None
    nodata: float | None = None
    bands: list[BandInfo] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding."""

    code: str
    message: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class ValidationResult:
    """Outcome of validating a file or a pair of rasters."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: RasterMetadata | None = None

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def add(self, code: str, message: str, severity: str = "error") -> None:
        self.issues.append(ValidationIssue(code=code, message=message, severity=severity))
        if severity == "error":
            self.valid = False


@dataclass
class CompatibilityResult:
    """Outcome of checking whether two rasters can be analyzed together."""

    compatible: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    overlap_ratio: float | None = None
    same_crs: bool | None = None
    resampling_required: bool = False
    alignment_required: bool = False
    notes: list[str] = field(default_factory=list)

    def add(self, code: str, message: str, severity: str = "error") -> None:
        self.issues.append(ValidationIssue(code=code, message=message, severity=severity))
        if severity == "error":
            self.compatible = False
