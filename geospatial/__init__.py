"""Remote-sensing raster IO, metadata, validation, alignment, and rendering.

Independent of the web layer so the same code serves the API and the ML pipelines.
All reads are windowed or decimated - see ``io.py``.
"""

from .alignment import check_pair_compatibility
from .bands import BandRoles, normalized_difference, resolve_band_roles
from .coords import box_to_geo, pixel_to_crs, to_lonlat
from .io import (
    DEFAULT_MAX_PREVIEW_SIDE,
    DEFAULT_TILE_SIZE,
    is_readable_raster,
    iter_tiles,
    read_decimated,
    read_window,
)
from .metadata import infer_modality, read_metadata
from .render import load_display_image, render_preview
from .resample import (
    ALIGN_RESAMPLING,
    DEFAULT_ALIGN_MAX_SIDE,
    AlignedModalities,
    AlignedPair,
    AlignmentError,
    align_modalities,
    align_pair,
)
from .types import (
    BandInfo,
    CompatibilityResult,
    Modality,
    RasterMetadata,
    ValidationIssue,
    ValidationResult,
)
from .validation import (
    GEOTIFF_EXTENSIONS,
    PLAIN_IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    validate_extension,
    validate_raster_file,
)

__all__ = [
    "ALIGN_RESAMPLING",
    "DEFAULT_ALIGN_MAX_SIDE",
    "DEFAULT_MAX_PREVIEW_SIDE",
    "DEFAULT_TILE_SIZE",
    "GEOTIFF_EXTENSIONS",
    "PLAIN_IMAGE_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "AlignedModalities",
    "AlignedPair",
    "AlignmentError",
    "BandInfo",
    "BandRoles",
    "CompatibilityResult",
    "Modality",
    "RasterMetadata",
    "ValidationIssue",
    "ValidationResult",
    "align_modalities",
    "align_pair",
    "box_to_geo",
    "check_pair_compatibility",
    "infer_modality",
    "is_readable_raster",
    "iter_tiles",
    "load_display_image",
    "normalized_difference",
    "pixel_to_crs",
    "read_decimated",
    "read_metadata",
    "read_window",
    "render_preview",
    "resolve_band_roles",
    "to_lonlat",
    "validate_extension",
    "validate_raster_file",
]
