---
inclusion: fileMatch
fileMatchPattern: 'geospatial/**'
---

# Geospatial Rules

Stack: Rasterio, GDAL, NumPy, OpenCV, scikit-image, pyproj, Shapely, GeoPandas.

## Correctness
- Always read and preserve **CRS, transform, bounds, resolution, band info, dtype,
  nodata**. Do not drop georeferencing silently.
- Coordinate conversions go through pyproj; never hand-roll reprojection math.
- Distinguish detection from interpretation; only report areas/percentages when
  geospatially valid (known transform/CRS).

## Large rasters (performance + safety)
- **Never load a huge GeoTIFF fully into RAM** unnecessarily. Use windowed reads and
  tiling/chunking. Downsample for previews.
- Cache derived previews/metadata; clean up temp artifacts.

## Pairs (temporal & optical-SAR)
- Validate before analysis: spatial overlap, CRS compatibility, extent, resolution,
  and required resampling/co-registration.
- **Do not run change/fusion analysis on clearly incompatible images.** Surface a
  clear validation error instead.
- Resampling must be explicit and recorded in the execution trace.

## Untrusted input
- Treat every uploaded raster as untrusted: validate format/MIME/corruption/dimensions
  before opening for processing. See `security.md`.
