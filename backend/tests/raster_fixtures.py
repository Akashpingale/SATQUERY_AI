"""Helpers that build small synthetic rasters for tests.

Real GeoTIFFs written by Rasterio - not mocks - so the geospatial code under test
exercises actual GDAL reading paths.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin


def write_geotiff(
    path: str | Path,
    *,
    width: int = 64,
    height: int = 64,
    bands: int = 3,
    dtype: str = "uint8",
    crs: str | None = "EPSG:4326",
    origin: tuple[float, float] = (77.0, 13.0),
    resolution: float = 0.001,
    nodata: float | None = None,
    tags: dict[str, str] | None = None,
    seed: int = 1234,
    array: np.ndarray | None = None,
    band_names: list[str] | None = None,
) -> Path:
    """Write a small deterministic GeoTIFF and return its path.

    ``seed`` exists so two fixtures can differ: temporal tests need T1 and T2 to be genuinely
    different images, and with a fixed seed every same-shaped fixture is pixel-identical,
    which a change detector would correctly measure as zero change.

    ``array`` supplies exact pixel content for a known-change pair. ``band_names`` sets GDAL
    band descriptions, which is what makes NDVI/NDWI computable - the code refuses to guess
    band order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if array is None:
        rng = np.random.default_rng(seed)
        array = (rng.random((bands, height, width)) * 200).astype(dtype)
    else:
        array = np.asarray(array)
        bands, height, width = array.shape

    profile: dict[str, object] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": bands,
        "dtype": dtype,
    }
    if crs is not None:
        profile["crs"] = crs
        profile["transform"] = from_origin(origin[0], origin[1], resolution, resolution)
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype))
        if band_names:
            for index, name in enumerate(band_names[:bands], start=1):
                dst.set_band_description(index, name)
        if tags:
            dst.update_tags(**tags)
    return path


def textured_scene(
    *,
    size: int = 96,
    bands: int = 3,
    base: float = 1200.0,
    noise: float = 30.0,
    seed: int = 7,
) -> np.ndarray:
    """A structured synthetic scene: smooth land-cover-like variation plus texture.

    Pure random noise is a poor stand-in for imagery here, because change detection measures
    structural correspondence between two dates. A scene with real spatial structure exercises
    the registration and coherence logic the way actual rasters do.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:size, 0:size]
    gradient = 320.0 * np.sin(cols / 19.0) + 260.0 * np.cos(rows / 23.0)
    planes = [
        np.clip(base + offset + gradient + rng.normal(0, noise, (size, size)), 1, 65535)
        for offset in (0.0, 130.0, -95.0)[:bands]
    ]
    return np.stack(planes)


def optical_sar_scene(
    *, size: int = 128, seed: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """A co-located optical/SAR pair with known, complementary features.

    Returns ``(optical (3,H,W) [Red,Green,NIR], sar (1,H,W))``:

    - a horizontal **water** strip: optically dark with collapsed NIR (NDWI-positive), and
      SAR-dark/smooth - the two modalities should *agree* on it;
    - a **structured block**: bright and rough in SAR but spectrally flat in optical - SAR
      should reveal structure the optical does not.

    This gives tests known ground truth for the cross-modal measurements.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:size, 0:size]
    red = np.clip(700 + 120 * np.sin(cols / 20) + rng.normal(0, 25, (size, size)), 1, 9000)
    green = np.clip(950 + rng.normal(0, 25, (size, size)), 1, 9000)
    nir = np.clip(3200 + 200 * np.cos(rows / 22) + rng.normal(0, 40, (size, size)), 1, 9000)

    water = slice(int(size * 0.70), int(size * 0.86))
    nir[water, :] = 300
    green[water, :] = 1100
    red[water, :] = 500
    optical = np.stack([red, green, nir])

    sar = np.clip(0.20 + rng.gamma(2.0, 0.05, (size, size)), 1e-3, 5.0)
    sar[water, :] = 0.03  # smooth dark water
    block = (slice(int(size * 0.16), int(size * 0.43)), slice(int(size * 0.20), int(size * 0.55)))
    sar[block] = np.clip(1.4 + rng.gamma(2.0, 0.2, sar[block].shape), 0, 6)
    return optical, sar[None, :, :]


def with_block(
    scene: np.ndarray,
    *,
    rows: tuple[int, int],
    cols: tuple[int, int],
    value: float | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Copy ``scene`` with a rectangular block replaced or rescaled.

    Returns the modified copy. The block's pixel count is exactly
    ``(rows[1] - rows[0]) * (cols[1] - cols[0])``, which lets a test assert the detected area
    against a known ground truth instead of a tolerance.
    """
    changed = scene.copy()
    window = (slice(None), slice(rows[0], rows[1]), slice(cols[0], cols[1]))
    if value is not None:
        changed[window] = value
    elif scale is not None:
        changed[window] = np.clip(changed[window] * scale, 1, 65535)
    return changed


def write_plain_png(path: str | Path, width: int = 32, height: int = 32) -> Path:
    """Write a non-georeferenced PNG (allowed for benchmark datasets)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    Image.fromarray((rng.random((height, width, 3)) * 255).astype("uint8"), "RGB").save(path)
    return path


def write_corrupt_tiff(path: str | Path) -> Path:
    """Write a file with a .tif extension whose contents are not a real raster."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"II*\x00 this is not a valid tiff body " + b"\x00" * 64)
    return path


def geotiff_bytes(**kwargs: object) -> bytes:
    """Return an in-memory GeoTIFF as bytes, for API upload tests."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = write_geotiff(Path(tmp) / "fixture.tif", **kwargs)  # type: ignore[arg-type]
        return path.read_bytes()


def png_bytes(width: int = 32, height: int = 32) -> bytes:
    rng = np.random.default_rng(7)
    buffer = io.BytesIO()
    Image.fromarray((rng.random((height, width, 3)) * 255).astype("uint8"), "RGB").save(
        buffer, format="PNG"
    )
    return buffer.getvalue()
