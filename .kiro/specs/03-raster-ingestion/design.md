# Spec 03 — Raster Ingestion · Design (as built, Phase 2)

## Overview
A shared `geospatial/` library plus a backend ingestion service and uploads API. Every
read is windowed or decimated, every upload is validated as a real raster before it is
trusted, and pair compatibility is decided on the server.

## Package layout (as built)
```
geospatial/                     installed as `satquery-geospatial` (root pyproject.toml)
├── types.py        Modality, BandInfo, RasterMetadata, ValidationIssue/Result, CompatibilityResult
├── metadata.py     read_metadata (header only), infer_modality (hint only)
├── io.py           read_window, iter_tiles, read_decimated, is_readable_raster
├── validation.py   validate_extension, validate_raster_file
├── alignment.py    check_pair_compatibility (pyproj reprojection)
└── render.py       render_preview (downsampled PNG)

backend/app/
├── security/files.py      sanitize_filename, build_storage_name, resolve_within, content-type allowlist
├── services/storage.py    storage root + subdirs, streaming size-limited writes
├── services/ingestion.py  ingest_upload, metadata_from_asset
├── models_db/raster_asset.py
├── schemas/uploads.py
└── api/v1/routes/uploads.py
```

`geospatial/` lives at the repository root (per the master plan layout) so the backend
and the ML pipelines share one implementation. It is installed via a root
`pyproject.toml` (`pip install -e .`).

## Memory safety
- `read_metadata` opens only the dataset header — safe on arbitrarily large rasters.
- `read_window` reads one clipped rectangle; out-of-range requests are clipped, not errors.
- `iter_tiles` streams tile-by-tile; peak memory ≈ `tile_size² × bands × dtype` (default 512).
- `read_decimated` uses Rasterio `out_shape` + `Resampling.average`, so downsampling
  happens *during* the read rather than after loading full resolution.
- Previews are capped at 1024 px on the longest side. The full-resolution raster is
  never sent to the browser.

## Validation order (cheapest first)
1. Extension allowlist (`.tif/.tiff/.gtiff/.geotiff`, plus `.png/.jpg/.jpeg` for benchmarks).
2. Existence and non-zero size.
3. Real open attempt via Rasterio — this is what catches corrupted files and files
   whose extension lies about their content.
4. Sanity limits: dimensions > 0, band count 1..512, pixel count ≤ 40000².
5. Georeferencing presence → reported as a **warning**, never an error (benchmark
   PNG/JPEG are legitimate inputs).

A raster is only treated as georeferenced when it has **both** a CRS and a
non-identity transform; otherwise `crs`/`bounds`/`resolution` are `None` rather than
Rasterio's identity-matrix default.

## Upload security
- Filenames are sanitized: directory components stripped (defeats `../` and absolute
  paths), unicode normalized, nulls removed, unsafe characters collapsed, length bounded,
  and unknown extensions dropped.
- Stored names are `<uuid12>_<safe-stem><ext>`, so collisions and overwrites are impossible.
- `resolve_within` re-resolves every path and rejects anything escaping the storage root.
- Writes stream in 1 MiB chunks against `MAX_UPLOAD_MB`; exceeding it deletes the
  partial file and returns 413.
- Declared content type is checked against an allowlist, but treated as advisory — the
  authoritative check is opening the file.
- **Order matters**: the file is streamed to disk under a safe name, then validated. If
  validation fails the file is deleted and no database row is created.

## Pair compatibility (`check_pair_compatibility`)
Decides whether two rasters may be analyzed together:
- **Non-georeferenced pairs**: overlap cannot be verified → warning + `alignment_required`.
  Identical pixel dimensions are treated as "assume pre-registered"; differing dimensions
  are a hard error (`dimension_mismatch`).
- **CRS mismatch**: warning, and image B's bounds are reprojected into A's CRS via pyproj
  before measuring overlap (so a UTM/WGS84 pair over the same ground is still recognized).
- **Overlap**: `< 30%` of the smaller footprint → `insufficient_overlap` (error);
  no intersection → `no_overlap` (error); `30–99%` → `partial_overlap` (warning).
- **Resolution**: ratio > 1.05 on either axis → `resolution_mismatch` + `resampling_required`.
- **Grid**: differing pixel dimensions → `alignment_required`.
- For `optical_sar` purpose the API additionally requires one optical **and** one SAR
  asset, otherwise `modality_mismatch`.

## API
```
POST   /api/v1/uploads                 multipart: file, role, modality -> 201 asset
GET    /api/v1/uploads                 recent assets
GET    /api/v1/uploads/{id}            metadata
GET    /api/v1/uploads/{id}/preview    PNG preview
DELETE /api/v1/uploads/{id}            remove asset + files
POST   /api/v1/uploads/compatibility   { asset_id_a, asset_id_b, purpose } -> verdict
```
Errors: 413 too large, 415 disallowed content type, 422 invalid raster/role/modality
(with structured `issues[]`), 404 unknown asset.

> These endpoints are currently **unauthenticated** — fine for local development and the
> demo, but authentication is required before any public deployment.

## Data model
`raster_assets`: identity, original + stored filename, storage/preview paths (relative
to the storage root), size, content type, role, `modality` (user-declared) kept separate
from `detected_modality` (inferred hint), dimensions/bands/dtype/driver, CRS + EPSG,
`is_georeferenced`, full `raster_metadata` JSON, and ingestion `warnings` JSON.

## Frontend
- `useUploads` manages the four slots (optical, SAR, T1, T2) and calls the backend for
  every compatibility verdict — the browser never decides compatibility itself.
- `UploadSlot`: drag-and-drop plus file picker on a real focusable `<button>`;
  status shown as icon + text; rejections render in a `role="alert"` block with the
  backend's plain-language issues.
- `MetadataPanel`: full metadata with an explicit "No georeferencing" state instead of
  invented coordinates; collapsible band details and metadata tags.
- `ImageViewer`: zoom (25–800%) / pan / fit over the server-rendered preview, with
  keyboard-operable toolbar controls and alt text.
- Degree-based resolution is labelled in degrees and **not** converted to metres, since
  that conversion is latitude-dependent.

## Decisions & rationale
- **Warning, not error, for missing georeferencing** — PNG/JPEG benchmark data is an
  allowed input; blocking it would break evaluation datasets.
- **Detected modality kept separate from declared modality** — a heuristic must never
  masquerade as a user statement or a measurement.
- **Compatibility on the server** — the same rules then apply to the agent (Phase 3+)
  and the UI, so there is one source of truth.
