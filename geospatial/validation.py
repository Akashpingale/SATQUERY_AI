"""Raster validation for untrusted uploads.

Checks are ordered cheapest-first: extension, then size, then a real attempt to open
the dataset header, then remote-sensing sanity checks. Nothing here reads pixel data.
"""

from __future__ import annotations

from pathlib import Path

import rasterio

from .metadata import read_metadata
from .types import RasterMetadata, ValidationResult

# Formats we accept. GeoTIFF/TIFF are the primary remote-sensing inputs; PNG/JPEG are
# permitted for public benchmark datasets that ship as plain images.
GEOTIFF_EXTENSIONS = {".tif", ".tiff", ".gtiff", ".geotiff"}
PLAIN_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = GEOTIFF_EXTENSIONS | PLAIN_IMAGE_EXTENSIONS

# Guardrails against absurd or hostile inputs (decompression bombs).
MAX_PIXELS = 40_000 * 40_000
MAX_BANDS = 512


def validate_extension(filename: str) -> ValidationResult:
    """Validate only the filename extension (pre-flight, before any file is written)."""
    result = ValidationResult(valid=True)
    suffix = Path(filename).suffix.lower()
    if not suffix:
        result.add(
            "missing_extension",
            "The file has no extension, so its format cannot be determined.",
        )
    elif suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        result.add(
            "unsupported_format",
            f"'{suffix}' is not a supported format. Supported formats: {supported}.",
        )
    return result


def validate_raster_file(
    path: str | Path, original_filename: str | None = None
) -> ValidationResult:
    """Fully validate a raster file on disk.

    Returns a :class:`ValidationResult` carrying metadata when the file is usable.
    Never raises for invalid input - problems are reported as issues so the API can
    explain them to the user in plain language.
    """
    path = Path(path)
    result = ValidationResult(valid=True)

    name_for_check = original_filename or path.name
    ext_result = validate_extension(name_for_check)
    result.issues.extend(ext_result.issues)
    if not ext_result.valid:
        result.valid = False
        return result

    if not path.exists():
        result.add("not_found", "The uploaded file could not be found on the server.")
        return result

    if path.stat().st_size == 0:
        result.add("empty_file", "The file is empty.")
        return result

    # Real format check: attempt to open the dataset. This catches corrupted files and
    # files whose extension does not match their actual content.
    try:
        metadata: RasterMetadata = read_metadata(path)
    except rasterio.errors.RasterioIOError:
        result.add(
            "unreadable_raster",
            "The file could not be opened as an image. It may be corrupted or not a real "
            "raster despite its extension.",
        )
        return result
    except Exception:  # noqa: BLE001 - any decode failure is a validation problem
        result.add("unreadable_raster", "The file could not be decoded as a raster image.")
        return result

    result.metadata = metadata

    if metadata.width <= 0 or metadata.height <= 0:
        result.add("invalid_dimensions", "The image reports zero or negative dimensions.")
        return result

    if metadata.band_count <= 0:
        result.add("no_bands", "The image contains no raster bands.")
        return result

    if metadata.band_count > MAX_BANDS:
        result.add(
            "too_many_bands",
            f"The image declares {metadata.band_count} bands, which exceeds the "
            f"limit of {MAX_BANDS}.",
        )

    if metadata.pixel_count > MAX_PIXELS:
        result.add(
            "raster_too_large",
            f"The image is {metadata.width}x{metadata.height} pixels, which exceeds the "
            "supported size limit.",
        )

    # Georeferencing is not required (benchmark PNG/JPEG are allowed) but its absence
    # limits geospatial output, so surface it as a warning rather than hiding it.
    if not metadata.is_georeferenced:
        result.add(
            "not_georeferenced",
            "The image has no CRS/transform. Analysis will work, but coordinates, areas, "
            "and pair alignment cannot be computed.",
            severity="warning",
        )

    return result
