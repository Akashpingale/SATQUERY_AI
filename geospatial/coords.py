"""Pixel-to-world coordinate conversion.

Used to turn model output in pixel space (e.g. a detection box) into real geographic
coordinates, which is what makes a detection on satellite imagery actionable rather than
merely visual. All CRS transformation goes through pyproj.

Every function returns ``None`` when the inputs cannot support a real answer, so callers
never present a fabricated coordinate.
"""

from __future__ import annotations

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

Transform6 = tuple[float, float, float, float, float, float]


def pixel_to_crs(transform: Transform6, col: float, row: float) -> tuple[float, float]:
    """Convert a pixel coordinate to CRS coordinates using an affine transform.

    ``transform`` is ``(a, b, c, d, e, f)`` in Rasterio/GDAL order, mapping
    ``x = a*col + b*row + c`` and ``y = d*col + e*row + f``.
    """
    a, b, c, d, e, f = transform
    x = a * col + b * row + c
    y = d * col + e * row + f
    return x, y


def to_lonlat(crs: str | None, x: float, y: float) -> tuple[float, float] | None:
    """Convert CRS coordinates to WGS84 longitude/latitude.

    Returns ``None`` when the CRS is missing or cannot be interpreted, rather than
    guessing that the values are already lon/lat.
    """
    if not crs:
        return None
    try:
        source = CRS.from_user_input(crs)
    except (CRSError, ValueError):
        return None

    if source.to_epsg() == 4326:
        return float(x), float(y)

    try:
        transformer = Transformer.from_crs(source, CRS.from_epsg(4326), always_xy=True)
        lon, lat = transformer.transform(x, y)
    except Exception:  # noqa: BLE001 - unusable CRS means no coordinate
        return None

    if lon != lon or lat != lat:  # NaN check
        return None
    return float(lon), float(lat)


def box_to_geo(
    box: tuple[float, float, float, float],
    transform: Transform6 | None,
    crs: str | None,
    scale: float = 1.0,
) -> dict[str, object] | None:
    """Convert a pixel-space box to geographic coordinates.

    Args:
        box: ``(x_min, y_min, x_max, y_max)`` in the coordinates of the *analyzed* image.
        transform: Affine transform of the full-resolution source raster.
        crs: CRS of the source raster.
        scale: Decimation factor applied to produce the analyzed image. Analyzed
            coordinates are divided by this to recover source pixel coordinates.

    Returns:
        A dict with source-pixel bounds, CRS bounds, and lon/lat centre where available,
        or ``None`` when the raster is not georeferenced.
    """
    if transform is None or not crs or scale <= 0:
        return None

    x_min, y_min, x_max, y_max = (value / scale for value in box)

    left, top = pixel_to_crs(transform, x_min, y_min)
    right, bottom = pixel_to_crs(transform, x_max, y_max)

    center_col = (x_min + x_max) / 2.0
    center_row = (y_min + y_max) / 2.0
    center_x, center_y = pixel_to_crs(transform, center_col, center_row)
    lonlat = to_lonlat(crs, center_x, center_y)

    payload: dict[str, object] = {
        "source_pixel_bounds": [
            round(x_min, 1),
            round(y_min, 1),
            round(x_max, 1),
            round(y_max, 1),
        ],
        "crs_bounds": [
            round(min(left, right), 6),
            round(min(top, bottom), 6),
            round(max(left, right), 6),
            round(max(top, bottom), 6),
        ],
        "crs": crs,
    }
    if lonlat is not None:
        payload["center_lonlat"] = [round(lonlat[0], 6), round(lonlat[1], 6)]
    return payload
