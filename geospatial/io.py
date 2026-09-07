"""Memory-safe raster reading.

All reads here are windowed or decimated. Nothing in this module loads a full-size
raster into RAM, which is what makes large GeoTIFF handling safe.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

# Cap for any single decimated read (pixels per side). Previews stay well under this.
DEFAULT_MAX_PREVIEW_SIDE = 1024
DEFAULT_TILE_SIZE = 512


def read_window(
    path: str | Path,
    row_off: int,
    col_off: int,
    height: int,
    width: int,
    bands: list[int] | None = None,
) -> np.ndarray:
    """Read a single rectangular window.

    Returns an array shaped ``(bands, height, width)``. The window is clipped to the
    dataset extent, so out-of-range requests are safe.
    """
    with rasterio.open(path) as src:
        row_off = max(0, min(row_off, src.height))
        col_off = max(0, min(col_off, src.width))
        height = max(0, min(height, src.height - row_off))
        width = max(0, min(width, src.width - col_off))
        window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
        indexes = bands or list(range(1, src.count + 1))
        return src.read(indexes=indexes, window=window)


def iter_tiles(
    path: str | Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    bands: list[int] | None = None,
) -> Iterator[tuple[Window, np.ndarray]]:
    """Iterate over a raster tile by tile.

    Yields ``(window, array)`` pairs so callers can process arbitrarily large rasters
    with bounded memory. Peak memory is roughly ``tile_size**2 * band_count * dtype``.
    """
    with rasterio.open(path) as src:
        indexes = bands or list(range(1, src.count + 1))
        for row_off in range(0, src.height, tile_size):
            for col_off in range(0, src.width, tile_size):
                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=min(tile_size, src.width - col_off),
                    height=min(tile_size, src.height - row_off),
                )
                yield window, src.read(indexes=indexes, window=window)


def read_decimated(
    path: str | Path,
    max_side: int = DEFAULT_MAX_PREVIEW_SIDE,
    bands: list[int] | None = None,
) -> tuple[np.ndarray, float]:
    """Read a whole raster downsampled so its longest side is <= ``max_side``.

    Uses Rasterio's ``out_shape`` decimation, which resamples during read instead of
    loading full resolution first. Returns ``(array, scale)`` where ``scale`` is the
    applied downsample factor (1.0 means no downsampling).
    """
    with rasterio.open(path) as src:
        longest = max(src.width, src.height)
        scale = 1.0 if longest <= max_side else max_side / longest
        out_h = max(1, int(round(src.height * scale)))
        out_w = max(1, int(round(src.width * scale)))
        indexes = bands or list(range(1, src.count + 1))
        data = src.read(
            indexes=indexes,
            out_shape=(len(indexes), out_h, out_w),
            resampling=Resampling.average,
        )
        return data, scale


def is_readable_raster(path: str | Path) -> bool:
    """Return True when the file can be opened as a raster (header check only)."""
    try:
        with rasterio.open(path) as src:
            return src.count > 0 and src.width > 0 and src.height > 0
    except Exception:  # noqa: BLE001 - any failure means "not a usable raster"
        return False
