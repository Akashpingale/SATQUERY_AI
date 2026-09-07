"""Raster metadata extraction.

Reads CRS, transform, bounds, resolution, band info, dtype, and nodata without
loading pixel data into memory. Georeferencing is reported honestly: plain images
(PNG/JPEG) yield ``is_georeferenced=False`` with ``crs``/``transform`` set to None
rather than fabricated defaults.
"""

from __future__ import annotations

from pathlib import Path

import rasterio

from .types import BandInfo, Modality, RasterMetadata


def read_metadata(path: str | Path) -> RasterMetadata:
    """Extract metadata from a raster file.

    Opens the dataset header only - no pixel data is read, so this is safe for very
    large rasters.

    Raises:
        rasterio.errors.RasterioIOError: if the file cannot be opened as a raster.
    """
    with rasterio.open(path) as src:
        crs_str: str | None = None
        crs_epsg: int | None = None
        if src.crs is not None:
            crs_str = src.crs.to_string()
            try:
                crs_epsg = src.crs.to_epsg()
            except Exception:  # noqa: BLE001 - malformed CRS should not break ingestion
                crs_epsg = None

        # A dataset is only meaningfully georeferenced when it has BOTH a CRS and a
        # real transform. Rasterio returns an identity transform for plain images.
        has_transform = src.transform is not None and not src.transform.is_identity
        is_geo = crs_str is not None and has_transform

        transform = None
        bounds = None
        resolution = None
        if has_transform:
            t = src.transform
            transform = (t.a, t.b, t.c, t.d, t.e, t.f)
            resolution = (abs(t.a), abs(t.e))
            if is_geo:
                b = src.bounds
                bounds = (b.left, b.bottom, b.right, b.top)

        bands = [
            BandInfo(
                index=i,
                dtype=str(src.dtypes[i - 1]),
                nodata=src.nodatavals[i - 1],
                description=src.descriptions[i - 1],
            )
            for i in range(1, src.count + 1)
        ]

        tags = {str(k): str(v) for k, v in (src.tags() or {}).items()}

        return RasterMetadata(
            width=src.width,
            height=src.height,
            band_count=src.count,
            dtype=str(src.dtypes[0]) if src.dtypes else "unknown",
            driver=src.driver,
            is_georeferenced=is_geo,
            crs=crs_str,
            crs_epsg=crs_epsg,
            transform=transform,
            bounds=bounds,
            resolution=resolution,
            nodata=src.nodata,
            bands=bands,
            tags=tags,
        )


def infer_modality(metadata: RasterMetadata) -> Modality:
    """Best-effort modality hint from metadata alone.

    This is a *hint* only - it is never used to override a user-declared modality.
    SAR products are commonly single-band with floating point or high-bit-depth
    amplitude/backscatter values, while optical products typically carry 3+ bands
    (RGB/multispectral) at 8-16 bit. Metadata tags mentioning a SAR sensor are the
    strongest signal.

    Returns ``Modality.UNKNOWN`` when the evidence is inconclusive, so callers must
    ask the user rather than guess.
    """
    tag_blob = " ".join(f"{k} {v}" for k, v in metadata.tags.items()).lower()
    sar_keywords = (
        "sar",
        "sentinel-1",
        "sentinel1",
        "risat",
        "radarsat",
        "backscatter",
        "vv",
        "vh",
    )
    if any(word in tag_blob for word in sar_keywords):
        return Modality.SAR

    optical_keywords = ("sentinel-2", "landsat", "resourcesat", "multispectral", "rgb")
    if any(word in tag_blob for word in optical_keywords):
        return Modality.OPTICAL

    if metadata.band_count >= 3:
        return Modality.OPTICAL
    if metadata.band_count == 1 and metadata.dtype in {"float32", "float64", "uint16", "int16"}:
        return Modality.SAR

    return Modality.UNKNOWN
