"""Co-registration: reading two rasters onto one common grid.

Comparing two dates pixel-for-pixel is only meaningful when both are on the same grid.
:func:`geospatial.alignment.check_pair_compatibility` decides *whether* a pair may be
compared at all; this module performs the comparison-ready read.

Two rules shape everything here:

1. **Never invent detail.** The common grid takes the *coarser* of the two resolutions, so
   the finer image is downsampled rather than the coarser one being upsampled into
   resolution it does not have.
2. **Say what was done.** Every operation applied - reprojection, resampling, cropping to
   the overlap, decimation, band truncation - is recorded in
   :attr:`AlignedPair.operations` so the execution trace can report it instead of silently
   transforming a user's data.

Reads go through Rasterio's ``WarpedVRT``, which warps during read. No full-resolution
array is ever materialised, and the output is capped by ``max_side``, so memory stays
bounded regardless of input size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from .alignment import _reproject_bounds
from .io import read_decimated
from .metadata import read_metadata
from .types import RasterMetadata

#: Longest side of the aligned arrays. 768 keeps a 3-band float32 pair near 27 MB, which is
#: comfortable on CPU while preserving enough detail for region-level change statistics.
DEFAULT_ALIGN_MAX_SIDE = 768

#: Resampling used for BOTH images. Symmetry matters more than the specific kernel here: if
#: the two dates were resampled differently, the difference between them would partly
#: reflect the resampling rather than real change. ``average`` also suppresses aliasing when
#: downsampling to the common grid.
ALIGN_RESAMPLING = Resampling.average


class AlignmentError(RuntimeError):
    """Raised when a pair cannot be placed on a common grid.

    Carries a stable ``code`` so the API can map it to a user-facing message without
    string matching.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class AlignedPair:
    """Two rasters read onto one common grid, ready for pixel-wise comparison."""

    first: np.ndarray  # (bands, height, width) float32
    second: np.ndarray  # (bands, height, width) float32
    #: True where the pixel carries real data in BOTH images.
    valid: np.ndarray  # (height, width) bool
    width: int
    height: int
    band_count: int
    #: Affine transform of the common grid, or None for non-georeferenced pairs.
    transform: tuple[float, float, float, float, float, float] | None
    crs: str | None
    #: Intersection bounds in ``crs``, when georeferenced.
    overlap_bounds: tuple[float, float, float, float] | None
    #: Ground area of one common-grid pixel in CRS units squared, when georeferenced.
    pixel_area: float | None
    #: Whether both inputs were georeferenced (so geographic claims are valid).
    georeferenced: bool
    #: Human-readable record of every operation applied, for the execution trace.
    operations: list[str] = field(default_factory=list)

    @property
    def valid_fraction(self) -> float:
        """Share of the common grid that carries data in both images."""
        total = self.valid.size
        return float(self.valid.sum() / total) if total else 0.0


@dataclass(frozen=True)
class AlignedModalities:
    """Two rasters of *different* modalities read onto one common grid.

    Unlike :class:`AlignedPair`, the two stacks keep their own band counts - fusion needs all
    of the optical bands (for spectral indices) and all of the SAR bands at once, so band
    truncation to a common count would throw away exactly the information being fused.
    """

    optical: np.ndarray  # (optical_bands, height, width) float32
    sar: np.ndarray  # (sar_bands, height, width) float32
    #: True where the pixel carries real data in BOTH modalities.
    valid: np.ndarray  # (height, width) bool
    width: int
    height: int
    optical_band_count: int
    sar_band_count: int
    transform: tuple[float, float, float, float, float, float] | None
    crs: str | None
    overlap_bounds: tuple[float, float, float, float] | None
    pixel_area: float | None
    georeferenced: bool
    operations: list[str] = field(default_factory=list)

    @property
    def valid_fraction(self) -> float:
        total = self.valid.size
        return float(self.valid.sum() / total) if total else 0.0


@dataclass(frozen=True)
class _GridPlan:
    """The common grid two rasters will be read onto, decided before any pixels are read."""

    crs: str
    transform: object  # rasterio Affine
    width: int
    height: int
    overlap_bounds: tuple[float, float, float, float]
    pixel_area: float


def _intersection(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])
    if right <= left or top <= bottom:
        return None
    return (left, bottom, right, top)


def _effective_resolution(
    metadata: RasterMetadata, bounds_in_target: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Pixel size of ``metadata`` expressed in the target CRS's units.

    Derived from the reprojected footprint rather than by converting the resolution
    directly, because a pixel size in degrees and one in metres are not comparable. This is
    an approximation over the footprint, which is what choosing a common grid needs.
    """
    width = max(1, metadata.width)
    height = max(1, metadata.height)
    span_x = abs(bounds_in_target[2] - bounds_in_target[0])
    span_y = abs(bounds_in_target[3] - bounds_in_target[1])
    return span_x / width, span_y / height


def _read_on_grid(
    path: str | Path,
    *,
    crs: str,
    transform,
    width: int,
    height: int,
    band_count: int,
    resampling: Resampling,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp-read ``band_count`` bands of ``path`` onto the given grid.

    Returns ``(data, valid)`` where ``data`` is ``(band_count, height, width)`` float32 and
    ``valid`` is a 2-D bool mask that is True only where every read band has real data.
    """
    indexes = list(range(1, band_count + 1))
    with rasterio.open(path) as src:
        with WarpedVRT(
            src,
            crs=crs,
            transform=transform,
            width=width,
            height=height,
            resampling=resampling,
        ) as vrt:
            data = vrt.read(indexes=indexes, out_dtype="float32")
            masks = vrt.read_masks(indexes=indexes)

    valid = np.all(masks > 0, axis=0) & np.all(np.isfinite(data), axis=0)
    return data, valid


def _validity_from_nodata(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """Validity mask for a decimated read, which carries no GDAL mask band."""
    valid = np.all(np.isfinite(data), axis=0)
    if nodata is not None:
        valid &= ~np.any(data == nodata, axis=0)
    return valid


def align_pair(
    path_a: str | Path,
    path_b: str | Path,
    *,
    max_side: int = DEFAULT_ALIGN_MAX_SIDE,
    resampling: Resampling = ALIGN_RESAMPLING,
    band_limit: int | None = None,
) -> AlignedPair:
    """Read two rasters onto one common grid.

    Georeferenced pairs are reprojected (if needed) and resampled to the coarser of the two
    resolutions, cropped to their overlap. Non-georeferenced pairs can only be compared when
    their pixel dimensions match, in which case they are assumed pre-registered and that
    assumption is recorded.

    Args:
        path_a: First (earlier) raster.
        path_b: Second (later) raster.
        max_side: Cap on the longest side of the returned arrays, bounding memory.
        resampling: Resampling applied identically to both images.
        band_limit: Optionally compare at most this many bands.

    Returns:
        An :class:`AlignedPair`.

    Raises:
        AlignmentError: when no common grid can be established.
    """
    meta_a = read_metadata(path_a)
    meta_b = read_metadata(path_b)

    band_count = min(meta_a.band_count, meta_b.band_count)
    if band_limit is not None:
        band_count = min(band_count, band_limit)
    if band_count < 1:
        raise AlignmentError("no_bands", "At least one readable band is required in each image.")

    operations: list[str] = []
    if meta_a.band_count != meta_b.band_count:
        operations.append(
            f"compared the first {band_count} band(s); the images carry "
            f"{meta_a.band_count} and {meta_b.band_count} bands"
        )

    both_georeferenced = meta_a.is_georeferenced and meta_b.is_georeferenced
    if both_georeferenced:
        return _align_georeferenced(
            path_a,
            path_b,
            meta_a,
            meta_b,
            band_count=band_count,
            max_side=max_side,
            resampling=resampling,
            operations=operations,
        )

    return _align_by_pixel_grid(
        path_a,
        path_b,
        meta_a,
        meta_b,
        band_count=band_count,
        max_side=max_side,
        operations=operations,
    )


def _plan_georeferenced_grid(
    meta_a: RasterMetadata,
    meta_b: RasterMetadata,
    *,
    max_side: int,
    resampling: Resampling,
    operations: list[str],
) -> _GridPlan:
    """Decide the common grid for two georeferenced rasters, before reading pixels.

    Shared by :func:`align_pair` and :func:`align_modalities` so the co-registration geometry
    (and the operations recorded for the trace) are identical however the pixels are then read.
    """
    if meta_a.bounds is None or meta_b.bounds is None or meta_a.resolution is None:
        raise AlignmentError(
            "missing_bounds", "Spatial bounds are unavailable for one of the images."
        )

    target_crs = str(meta_a.crs)
    bounds_b = meta_b.bounds
    if meta_a.crs != meta_b.crs:
        reprojected = _reproject_bounds(meta_b.bounds, str(meta_b.crs), target_crs)
        if reprojected is None:
            raise AlignmentError(
                "crs_unusable",
                "The coordinate reference system of one image could not be interpreted, so "
                "the pair cannot be placed on a common grid.",
            )
        bounds_b = reprojected
        operations.append(f"reprojected the second image from {meta_b.crs} to {target_crs}")

    overlap = _intersection(meta_a.bounds, bounds_b)
    if overlap is None:
        raise AlignmentError(
            "no_overlap",
            "The images do not overlap geographically, so there is no shared area to compare.",
        )

    # Coarser resolution wins, so no image is upsampled into detail it does not have.
    res_b = _effective_resolution(meta_b, bounds_b)
    xres = max(meta_a.resolution[0], res_b[0])
    yres = max(meta_a.resolution[1], res_b[1])
    if xres <= 0 or yres <= 0:
        raise AlignmentError(
            "bad_resolution", "One of the images reports a non-positive pixel size."
        )
    if (xres, yres) != (meta_a.resolution[0], meta_a.resolution[1]):
        operations.append(
            f"used the coarser pixel size ({xres:g} x {yres:g}) so neither image is upsampled"
        )

    left, bottom, right, top = overlap
    width = int(np.floor((right - left) / xres))
    height = int(np.floor((top - bottom) / yres))
    if width < 2 or height < 2:
        raise AlignmentError(
            "overlap_too_small",
            "The overlapping area is smaller than two pixels on the common grid, which is "
            "too little to compare.",
        )

    longest = max(width, height)
    if longest > max_side:
        factor = longest / max_side
        xres *= factor
        yres *= factor
        width = max(2, int(np.floor((right - left) / xres)))
        height = max(2, int(np.floor((top - bottom) / yres)))
        operations.append(
            f"decimated to {width} x {height} px (longest side {max_side}) to bound memory"
        )

    transform = from_origin(left, top, xres, yres)
    operations.append(
        f"resampled both images onto a shared {width} x {height} grid using "
        f"{resampling.name} resampling"
    )

    footprint_a = (meta_a.bounds[2] - meta_a.bounds[0]) * (meta_a.bounds[3] - meta_a.bounds[1])
    overlap_area = (right - left) * (top - bottom)
    if footprint_a > 0 and overlap_area < 0.99 * footprint_a:
        operations.append(
            f"cropped to the shared area ({overlap_area / footprint_a:.0%} of the first image)"
        )

    return _GridPlan(
        crs=target_crs,
        transform=transform,
        width=width,
        height=height,
        overlap_bounds=(left, bottom, right, top),
        pixel_area=float(xres * yres),
    )


def _align_georeferenced(
    path_a: str | Path,
    path_b: str | Path,
    meta_a: RasterMetadata,
    meta_b: RasterMetadata,
    *,
    band_count: int,
    max_side: int,
    resampling: Resampling,
    operations: list[str],
) -> AlignedPair:
    plan = _plan_georeferenced_grid(
        meta_a, meta_b, max_side=max_side, resampling=resampling, operations=operations
    )

    data_a, valid_a = _read_on_grid(
        path_a,
        crs=plan.crs,
        transform=plan.transform,
        width=plan.width,
        height=plan.height,
        band_count=band_count,
        resampling=resampling,
    )
    data_b, valid_b = _read_on_grid(
        path_b,
        crs=plan.crs,
        transform=plan.transform,
        width=plan.width,
        height=plan.height,
        band_count=band_count,
        resampling=resampling,
    )

    valid = valid_a & valid_b
    transform = plan.transform
    affine = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)

    return AlignedPair(
        first=data_a,
        second=data_b,
        valid=valid,
        width=plan.width,
        height=plan.height,
        band_count=band_count,
        transform=affine,
        crs=plan.crs,
        overlap_bounds=plan.overlap_bounds,
        pixel_area=plan.pixel_area,
        georeferenced=True,
        operations=operations,
    )


def align_modalities(
    optical_path: str | Path,
    sar_path: str | Path,
    *,
    max_side: int = DEFAULT_ALIGN_MAX_SIDE,
    resampling: Resampling = ALIGN_RESAMPLING,
) -> AlignedModalities:
    """Read an optical and a SAR raster onto one common grid, keeping each band set intact.

    The common grid is chosen exactly as for :func:`align_pair` (optical's CRS, the coarser
    resolution, cropped to the overlap), so no modality is upsampled and every operation is
    recorded. Unlike ``align_pair`` this keeps **all** optical bands and **all** SAR bands,
    because fusion needs the optical spectral bands and the SAR backscatter together.

    Raises:
        AlignmentError: when no common grid can be established, or when a modality has no
            readable band.
    """
    meta_o = read_metadata(optical_path)
    meta_s = read_metadata(sar_path)
    if meta_o.band_count < 1 or meta_s.band_count < 1:
        raise AlignmentError("no_bands", "Each modality needs at least one readable band.")

    operations: list[str] = []

    if meta_o.is_georeferenced and meta_s.is_georeferenced:
        plan = _plan_georeferenced_grid(
            meta_o, meta_s, max_side=max_side, resampling=resampling, operations=operations
        )
        optical, valid_o = _read_on_grid(
            optical_path, crs=plan.crs, transform=plan.transform, width=plan.width,
            height=plan.height, band_count=meta_o.band_count, resampling=resampling,
        )
        sar, valid_s = _read_on_grid(
            sar_path, crs=plan.crs, transform=plan.transform, width=plan.width,
            height=plan.height, band_count=meta_s.band_count, resampling=resampling,
        )
        transform = plan.transform
        affine = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
        return AlignedModalities(
            optical=optical,
            sar=sar,
            valid=valid_o & valid_s,
            width=plan.width,
            height=plan.height,
            optical_band_count=meta_o.band_count,
            sar_band_count=meta_s.band_count,
            transform=affine,
            crs=plan.crs,
            overlap_bounds=plan.overlap_bounds,
            pixel_area=plan.pixel_area,
            georeferenced=True,
            operations=operations,
        )

    # Fallback: no georeferencing on one side. Only equal pixel grids can be paired.
    if (meta_o.width, meta_o.height) != (meta_s.width, meta_s.height):
        raise AlignmentError(
            "dimension_mismatch",
            f"The optical ({meta_o.width}x{meta_o.height}) and SAR "
            f"({meta_s.width}x{meta_s.height}) images have different pixel dimensions and lack "
            "the georeferencing needed to align them.",
        )
    operations.append(
        "at least one image lacks georeferencing; the pair was compared on its shared pixel "
        "grid and geographic coordinates are not reported"
    )
    optical, _ = read_decimated(optical_path, max_side=max_side)
    sar, _ = read_decimated(sar_path, max_side=max_side)
    if optical.shape[1:] != sar.shape[1:]:
        raise AlignmentError(
            "dimension_mismatch",
            "The images could not be read onto a matching grid despite equal dimensions.",
        )
    optical = optical.astype("float32", copy=False)
    sar = sar.astype("float32", copy=False)
    height, width = optical.shape[1], optical.shape[2]
    if max(meta_o.width, meta_o.height) > max_side:
        operations.append(f"decimated to {width} x {height} px to bound memory")
    valid = _validity_from_nodata(optical, meta_o.nodata) & _validity_from_nodata(
        sar, meta_s.nodata
    )
    return AlignedModalities(
        optical=optical,
        sar=sar,
        valid=valid,
        width=width,
        height=height,
        optical_band_count=optical.shape[0],
        sar_band_count=sar.shape[0],
        transform=None,
        crs=None,
        overlap_bounds=None,
        pixel_area=None,
        georeferenced=False,
        operations=operations,
    )


def _align_by_pixel_grid(
    path_a: str | Path,
    path_b: str | Path,
    meta_a: RasterMetadata,
    meta_b: RasterMetadata,
    *,
    band_count: int,
    max_side: int,
    operations: list[str],
) -> AlignedPair:
    """Fallback for pairs that cannot be warped: identical pixel grids only."""
    if (meta_a.width, meta_a.height) != (meta_b.width, meta_b.height):
        raise AlignmentError(
            "dimension_mismatch",
            f"The images have different pixel dimensions ({meta_a.width}x{meta_a.height} vs "
            f"{meta_b.width}x{meta_b.height}) and lack the georeferencing needed to align them.",
        )

    if meta_a.is_georeferenced != meta_b.is_georeferenced:
        operations.append(
            "one image lacks georeferencing, so the pair was compared on its pixel grid and "
            "geographic coordinates are not reported"
        )
    else:
        operations.append(
            "no georeferencing available; the images share pixel dimensions and were compared "
            "as pre-registered"
        )

    indexes = list(range(1, band_count + 1))
    data_a, _ = read_decimated(path_a, max_side=max_side, bands=indexes)
    data_b, _ = read_decimated(path_b, max_side=max_side, bands=indexes)

    if data_a.shape != data_b.shape:
        raise AlignmentError(
            "dimension_mismatch",
            "The images could not be read onto a matching grid despite equal dimensions.",
        )

    height, width = data_a.shape[1], data_a.shape[2]
    if max(meta_a.width, meta_a.height) > max_side:
        operations.append(f"decimated to {width} x {height} px to bound memory")

    data_a = data_a.astype("float32", copy=False)
    data_b = data_b.astype("float32", copy=False)
    valid = _validity_from_nodata(data_a, meta_a.nodata) & _validity_from_nodata(
        data_b, meta_b.nodata
    )

    return AlignedPair(
        first=data_a,
        second=data_b,
        valid=valid,
        width=width,
        height=height,
        band_count=band_count,
        transform=None,
        crs=None,
        overlap_bounds=None,
        pixel_area=None,
        georeferenced=False,
        operations=operations,
    )
