"""Pair compatibility and alignment checks.

Used before any temporal-change or optical-SAR fusion analysis. The goal is to refuse
clearly incompatible pairs loudly instead of producing a confident-looking but
meaningless result. All coordinate transformation goes through pyproj.
"""

from __future__ import annotations

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

from .types import CompatibilityResult, RasterMetadata

# A pair whose overlap falls below this fraction of the smaller footprint is treated as
# not describing the same area.
MIN_OVERLAP_RATIO = 0.30
# Resolution ratio beyond which resampling is required before joint analysis.
MAX_RESOLUTION_RATIO = 1.05
# Shape mismatch alone does not block analysis, but it requires alignment.


def _reproject_bounds(
    bounds: tuple[float, float, float, float], src_crs: str, dst_crs: str
) -> tuple[float, float, float, float] | None:
    """Reproject a bounding box between CRSs. Returns None if the CRS is unusable."""
    try:
        transformer = Transformer.from_crs(
            CRS.from_user_input(src_crs),
            CRS.from_user_input(dst_crs),
            always_xy=True,
        )
    except (CRSError, ValueError):
        return None

    left, bottom, right, top = bounds
    xs, ys = transformer.transform(
        [left, right, left, right],
        [bottom, top, top, bottom],
    )
    finite = [
        (x, y)
        for x, y in zip(xs, ys, strict=True)
        if x == x and y == y and abs(x) != float("inf") and abs(y) != float("inf")
    ]
    if len(finite) < 2:
        return None
    xs_f = [p[0] for p in finite]
    ys_f = [p[1] for p in finite]
    return (min(xs_f), min(ys_f), max(xs_f), max(ys_f))


def _intersection_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def check_pair_compatibility(
    a: RasterMetadata,
    b: RasterMetadata,
    label_a: str = "first image",
    label_b: str = "second image",
) -> CompatibilityResult:
    """Check whether two rasters describe the same area closely enough to analyze jointly.

    Reports spatial overlap, CRS compatibility, resolution differences, and whether
    resampling or alignment is required. Non-georeferenced pairs cannot be verified
    spatially: that is reported as a warning plus an explicit alignment requirement,
    never silently assumed to be fine.
    """
    result = CompatibilityResult(compatible=True)

    # --- Non-georeferenced inputs -------------------------------------------------
    if not a.is_georeferenced or not b.is_georeferenced:
        which = []
        if not a.is_georeferenced:
            which.append(label_a)
        if not b.is_georeferenced:
            which.append(label_b)
        result.add(
            "not_georeferenced",
            f"Spatial overlap cannot be verified because {' and '.join(which)} "
            "lack a CRS/transform.",
            severity="warning",
        )
        if (a.width, a.height) != (b.width, b.height):
            result.add(
                "dimension_mismatch",
                f"The images have different pixel dimensions ({a.width}x{a.height} vs "
                f"{b.width}x{b.height}) and no georeferencing to align them with.",
            )
        else:
            result.notes.append(
                "Images share identical pixel dimensions; assuming they are pre-registered."
            )
        result.same_crs = None
        result.alignment_required = True
        return result

    # --- CRS ----------------------------------------------------------------------
    result.same_crs = a.crs == b.crs
    if not result.same_crs:
        result.add(
            "crs_mismatch",
            f"The images use different coordinate reference systems ({a.crs} vs {b.crs}). "
            "Reprojection is required before joint analysis.",
            severity="warning",
        )
        result.alignment_required = True

    # --- Spatial overlap (compare in image A's CRS) --------------------------------
    bounds_a = a.bounds
    bounds_b = b.bounds
    if bounds_a is None or bounds_b is None:
        result.add("missing_bounds", "Spatial bounds are unavailable for one of the images.")
        return result

    if not result.same_crs:
        reprojected = _reproject_bounds(bounds_b, str(b.crs), str(a.crs))
        if reprojected is None:
            result.add(
                "crs_unusable",
                "The coordinate reference system of one image could not be interpreted, so "
                "spatial overlap cannot be computed.",
            )
            return result
        bounds_b = reprojected

    intersection = _intersection_area(bounds_a, bounds_b)
    smaller = min(_area(bounds_a), _area(bounds_b))
    ratio = (intersection / smaller) if smaller > 0 else 0.0
    result.overlap_ratio = round(ratio, 4)

    if intersection <= 0:
        result.add(
            "no_overlap",
            "The images do not overlap geographically, so they cannot describe the same area.",
        )
        return result

    if ratio < MIN_OVERLAP_RATIO:
        result.add(
            "insufficient_overlap",
            f"The images overlap by only {ratio:.0%} of the smaller footprint, which is too "
            "little for reliable joint analysis.",
        )
    elif ratio < 0.99:
        result.add(
            "partial_overlap",
            f"The images overlap by {ratio:.0%}. Analysis will be limited to the shared area.",
            severity="warning",
        )
        result.alignment_required = True

    # --- Resolution ----------------------------------------------------------------
    if a.resolution and b.resolution:
        ax, ay = a.resolution
        bx, by = b.resolution
        for axis, (ra, rb) in (("x", (ax, bx)), ("y", (ay, by))):
            lo, hi = sorted((abs(ra), abs(rb)))
            if lo <= 0:
                continue
            if hi / lo > MAX_RESOLUTION_RATIO:
                result.resampling_required = True
                result.add(
                    "resolution_mismatch",
                    f"The images have different {axis} resolutions ({ra:g} vs {rb:g}). "
                    "Resampling to a common grid is required.",
                    severity="warning",
                )

    # --- Grid alignment -------------------------------------------------------------
    if (a.width, a.height) != (b.width, b.height):
        result.alignment_required = True
        result.notes.append(
            f"Pixel grids differ ({a.width}x{a.height} vs {b.width}x{b.height}); "
            "the pair will be aligned to a common grid."
        )

    if result.compatible and not result.issues:
        result.notes.append("Images are co-registered and directly comparable.")

    return result
