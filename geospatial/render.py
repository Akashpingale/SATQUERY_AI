"""Browser-friendly preview rendering.

Converts a raster (any band count, any dtype) into a downsampled 8-bit PNG suitable
for display. Rendering is a *visualization* concern only - it never alters stored data
and never claims to be analysis output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .io import DEFAULT_MAX_PREVIEW_SIDE, read_decimated
from .metadata import read_metadata


def _stretch_to_uint8(band: np.ndarray, nodata: float | None = None) -> np.ndarray:
    """Percentile-stretch a single band to 0-255 for display.

    A 2-98 percentile stretch is used because raw remote-sensing values (16-bit
    optical, floating point SAR backscatter) are otherwise nearly black on screen.
    """
    data = band.astype("float64", copy=True)

    mask = ~np.isfinite(data)
    if nodata is not None:
        mask |= data == nodata
    valid = data[~mask]

    if valid.size == 0:
        return np.zeros(data.shape, dtype="uint8")

    lo, hi = np.percentile(valid, (2.0, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return np.zeros(data.shape, dtype="uint8")

    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0) * 255.0
    scaled[mask] = 0
    return scaled.astype("uint8")


def _decimated_display_array(
    source: str | Path, max_side: int
) -> tuple[np.ndarray, float, bool]:
    """Read a decimated, display-stretched array.

    Returns ``(array, scale, is_rgb)`` where ``array`` is ``(H, W, 3)`` for RGB sources
    or ``(H, W)`` for single-band sources, and ``scale`` is the decimation factor applied
    (multiply a source pixel coordinate by ``scale`` to get the array coordinate).
    """
    metadata = read_metadata(source)
    band_indexes = [1, 2, 3] if metadata.band_count >= 3 else [1]

    data, scale = read_decimated(source, max_side=max_side, bands=band_indexes)

    if data.shape[0] >= 3:
        channels = [_stretch_to_uint8(data[i], metadata.nodata) for i in range(3)]
        return np.dstack(channels), scale, True
    return _stretch_to_uint8(data[0], metadata.nodata), scale, False


def render_preview(
    source: str | Path,
    destination: str | Path,
    max_side: int = DEFAULT_MAX_PREVIEW_SIDE,
) -> tuple[int, int]:
    """Render a downsampled PNG preview of ``source`` at ``destination``.

    Band selection: 3+ bands render as RGB from the first three bands; 2 bands and
    single-band rasters (typical for SAR) render as grayscale. Reading is decimated,
    so memory stays bounded regardless of the source size.

    Returns the ``(width, height)`` of the written preview.
    """
    array, _scale, is_rgb = _decimated_display_array(source, max_side)
    image = Image.fromarray(array, mode="RGB" if is_rgb else "L")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return image.width, image.height


def load_display_image(
    source: str | Path,
    max_side: int = DEFAULT_MAX_PREVIEW_SIDE,
) -> tuple[Image.Image, float]:
    """Load a raster as a downsampled 3-channel RGB image for model input.

    Vision models expect 8-bit RGB, so single-band rasters (typical for SAR) are
    replicated across three channels rather than being rejected. The same percentile
    stretch used for previews is applied, so what a model sees matches what the user
    sees in the UI.

    Returns ``(image, scale)``. Divide an image coordinate by ``scale`` to recover the
    coordinate in the full-resolution source raster.
    """
    array, scale, is_rgb = _decimated_display_array(source, max_side)
    if not is_rgb:
        array = np.stack([array] * 3, axis=-1)
    return Image.fromarray(array, mode="RGB"), scale
