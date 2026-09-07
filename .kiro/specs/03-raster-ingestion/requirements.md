# Spec 03 — Raster Ingestion (stub)

> Phase 2. Stub now; full `design.md` + `tasks.md` authored when the phase begins.

## Introduction
Ingest, validate, and describe remote-sensing rasters (GeoTIFF/TIFF, plus PNG/JPEG
for benchmark data) safely and at scale, producing metadata and browser-friendly
previews for the UI and downstream specialists.

## Requirements (high level)
- Validate uploads (extension/MIME/size/corruption) and reject unsafe/unsupported files.
- Extract metadata: dimensions, band count, dtype, nodata, CRS, transform, bounds, resolution.
- Windowed/tiled reading; downsampled previews; never load huge rasters fully into RAM.
- Pair validation for temporal and optical+SAR: overlap, CRS, extent, resolution, resampling needs.

## Design (outline)
`geospatial/` modules: `io.py`, `metadata.py`, `validation.py`, `alignment.py`,
`tiling.py`, `render.py` (see `geospatial/README.md`). Backend `raster/` service wraps
these and persists metadata via the DB; artifacts to `storage/`.

## Implementation tasks (outline)
Readers → metadata extractor → validators → preview renderer → pair/alignment checks →
API integration (uploads) → tests + fixtures.

## Dependencies
Rasterio, GDAL, NumPy, OpenCV, scikit-image, pyproj, Shapely, GeoPandas (added this
phase). Phase 1 backend + storage.

## Acceptance criteria
- Valid GeoTIFF returns correct metadata; corrupted/unsupported files are rejected clearly.
- Incompatible pairs are flagged, not silently analyzed.
- Large rasters are handled via windows/tiles within memory limits.

## Tests
Unit: metadata/CRS/validation/alignment with small fixtures. Negative: corrupted TIFF,
unsupported format, mismatched CRS, non-overlapping pair.

## User inputs required
- Small sample GeoTIFF/SAR fixtures (can be synthesized; ask user if official samples needed).

## Known risks
- GDAL/Rasterio install on Windows (use wheels/conda or run in Docker).
- SAR formats vary; document supported subset.
