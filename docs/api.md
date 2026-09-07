# API Reference — SatQuery AI

Base path: `/api/v1`. Interactive docs (OpenAPI): `http://localhost:8000/docs`.

This lists **implemented** endpoints. Analysis, trace, evidence, history, and the
downloadable report are all live (Phases 3–8).

> **Security note:** the API is currently **unauthenticated**. That is acceptable for
> local development and the evaluation demo, but authentication and authorization must be
> added before any public deployment.

## System

### `GET /api/v1/health`
```json
{
  "status": "ok",
  "app": "SatQuery AI",
  "version": "0.1.0",
  "environment": "development",
  "device": "cpu",
  "database": { "connected": true, "dialect": "sqlite" }
}
```
`status` is `degraded` when the database is unreachable.

### `GET /api/v1/models`
Returns the model/tool registry: `{ count, tools[], health{} }`. Each tool declares
`id, name, version, task, modalities, input_contract, output_contract,
allowed_parameters, device, checkpoint, license, status, phase`.

`status` is one of `available | degraded | planned | unavailable` and is kept honest —
a specialist is only `available` when a real implementation exists.

### `GET /api/v1/tasks`
Returns supported task types with `min_images` / `max_images` / `modalities` / `phase`.

## Uploads

### `POST /api/v1/uploads`
`multipart/form-data`:

| Field | Type | Notes |
|---|---|---|
| `file` | file | GeoTIFF/TIFF, or PNG/JPEG for benchmark data |
| `role` | string | `single \| optical \| sar \| t1 \| t2` |
| `modality` | string | `optical \| sar \| unknown` |

**201** → `{ "asset": RasterAsset }`

`RasterAsset` includes `id, original_filename, size_bytes, role, modality,
detected_modality, width, height, band_count, dtype, driver, georeference{}, bands[],
nodata, tags{}, preview_url, warnings[]`.

`georeference` is `{ is_georeferenced, crs, crs_epsg, transform, bounds, resolution }`;
every field except `is_georeferenced` is `null` for non-georeferenced images.

Errors:

| Status | Cause |
|---|---|
| 413 | Exceeds `MAX_UPLOAD_MB` |
| 415 | Content type not in the allowlist |
| 422 | Not a usable raster, or invalid `role`/`modality`. Body carries `detail.issues[]` with `code`, `message`, `severity` |

Common issue codes: `unsupported_format`, `missing_extension`, `empty_file`,
`unreadable_raster`, `invalid_dimensions`, `no_bands`, `too_many_bands`,
`raster_too_large`, `not_georeferenced` (warning).

### `GET /api/v1/uploads`
`{ count, assets[] }` — most recent first. Query: `limit` (1–200, default 50).

### `GET /api/v1/uploads/{id}`
A single `RasterAsset`. **404** if unknown.

### `GET /api/v1/uploads/{id}/preview`
`image/png` — server-rendered, downsampled preview (≤1024 px longest side).
**404** when no preview exists.

### `DELETE /api/v1/uploads/{id}`
**204**. Removes the database row, the stored raster, and the preview.

### `POST /api/v1/uploads/compatibility`
```json
{ "asset_id_a": "...", "asset_id_b": "...", "purpose": "temporal" }
```
`purpose`: `temporal` or `optical_sar`.

```json
{
  "compatible": true,
  "purpose": "temporal",
  "overlap_ratio": 1.0,
  "same_crs": true,
  "resampling_required": false,
  "alignment_required": false,
  "issues": [],
  "notes": ["Images are co-registered and directly comparable."],
  "asset_a": { "id": "...", "modality": "optical" },
  "asset_b": { "id": "...", "modality": "optical" }
}
```

This is the guard that prevents change or fusion analysis from running on images that do
not describe the same area. Issue codes: `no_overlap`, `insufficient_overlap` (<30%),
`partial_overlap` (warning), `crs_mismatch` (warning), `resolution_mismatch` (warning),
`dimension_mismatch`, `not_georeferenced` (warning), `modality_mismatch`
(optical-SAR requires one of each).

`overlap_ratio` and `same_crs` are `null` when the inputs are not georeferenced, meaning
overlap could not be verified — not that it is fine.

**422** when both ids are identical; **404** when either asset is unknown.

## Errors
Unhandled failures return a sanitized envelope (no stack traces, no secrets):
```json
{ "error": { "code": "internal_error", "message": "An internal error occurred.", "request_id": "abc123" } }
```
Every response carries an `X-Request-ID` header for log correlation.

## Analysis

### `POST /api/v1/analysis`
Run the agent pipeline over one or two uploaded rasters.

```json
{ "query": "What CRS and resolution does this image use?", "asset_ids": ["<id>"] }
```
`asset_ids`: 1–2 unique ids. `query`: 1–2000 characters.

**201** → an analysis envelope. Note that 201 is returned for *all* outcomes; the `status`
field carries the result, so the UI can show the reasoning, trace, and partial evidence
instead of a bare HTTP error.

| `status` | Meaning | `answer` | `confidence.level` |
|---|---|---|---|
| `completed` | a genuine result was produced | text | `High`/`Medium`/`Low` with a score |
| `rejected` | inputs or query cannot be analyzed | reason text, or `null` | usually `Unable to determine` |
| `unavailable` | task understood and routed, specialist not built yet | `null` | `Unable to determine`, score `null` |
| `failed` | unexpected execution error | `null` | `Unable to determine` |

```json
{
  "id": "…", "created_at": "…",
  "status": "completed",
  "task": "metadata_inspection",
  "query": "What CRS and resolution does this image use?",
  "answer": "scene.tif is a 512 x 512 pixel GTiff with 3 band(s) of type uint8, georeferenced in EPSG:4326, …",
  "confidence": {
    "level": "High",
    "score": 0.949,
    "signals": ["Specialist result strength 1.00", "All inputs georeferenced", "…"]
  },
  "evidence": [
    { "kind": "metadata", "label": "Metadata for scene.tif", "asset_id": "…", "detail": {"crs_epsg": 4326}, "reference": null },
    { "kind": "preview", "label": "Preview of scene.tif", "asset_id": "…", "detail": {}, "reference": "/api/v1/uploads/…/preview" }
  ],
  "trace": [ { "step": "Task classification", "status": "ok", "detail": "Classified as metadata_inspection (routing score 0.86)" } ],
  "plan": { "task": "metadata_inspection", "steps": [ { "order": 1, "tool_id": "raster_preprocessor", "tool_version": "1.0.0", "description": "…", "parameters": {"max_preview_side": 1024} } ] },
  "issues": [], "warnings": [], "input_asset_ids": ["…"]
}
```

**Confidence** is always derived from listed signals. When no result strength is available
the level is `Unable to determine` and `score` is `null` — a value is never invented.

**Evidence** `kind` values:

| Kind | Meaning | Key `detail` fields |
|---|---|---|
| `metadata` | Measured raster metadata | dimensions, CRS, resolution, bounds |
| `preview` | Server-rendered image preview | — (`reference` is the image URL) |
| `overlay` | Rendered image with boxes/regions drawn | `box_count`, `method` |
| `box` | A detected bounding box | `score`, `analyzed_pixel_box`, `geo.center_lonlat` |
| `region` | A coarse tile-level region (not an object outline) | `tile_contrast`, `tiles_merged`, `grid`, `geo` |
| `scores` | Ranked model concept probabilities | `concepts[]`, `model_id`, `note` |
| `model` | Provenance of the model/method that ran | `model_id`/`algorithm_version`, `device`, `note` |
| `comparison` | Pair-compatibility assessment | `compatible`, `overlap_ratio`, `issues` |
| `change_map` | Temporal change map (image) | `changed_percent`, `region_count`, `threshold`, `co_registration` |
| `temporal_compare` | T1/T2/change strip (image) | `panels`, `note` |
| `change_region` | A coherent changed area | `pixel_count`, `area_m2`, `centroid_lonlat`, `position` |
| `indices` | NDVI/NDWI change (or why unavailable) | `indices[]` (`available`, `delta`, `reason`) |
| `fusion_composite` | Optical+SAR false-colour composite (image) | `correlation`, `complementarity`, `note` |
| `fusion_complementary` | Complementary-information map (image) | `optical_only_information`, `sar_only_information`, `shared_information` |
| `aligned_optical` / `aligned_sar` | The two modalities on the shared grid (image) | `bands`, `grid`, `representation` |
| `modality_comparison` | Optical vs SAR measured features | `optical{}`, `sar{}`, `correlation`, `sar_representation` |
| `cross_modal_water` | Optical/SAR water agreement | `agreement_iou`, `agree_fraction`, `sar_only_fraction` |
| `complementary_region` | A region one modality reveals | `kind`, `pixel_count`, `area_m2`, `centroid_lonlat` |

`reference` is an API path for image evidence. Geographic fields are present only for
georeferenced rasters; they are omitted rather than fabricated otherwise.

### `GET /api/v1/evidence/{token}`
Serves a rendered evidence image (e.g. a grounding overlay) as `image/png`. The token is a
server-generated hex identifier; the resolved path is confined to the evidence directory, so
this cannot read arbitrary files. **404** for unknown or malformed tokens.

### Model availability
Single-image AI runs on cached local weights. If weights cannot be loaded, the analysis
returns `status: "unavailable"` with issue code `model_unavailable` and `answer: null` — no
result is invented. See `docs/models.md`.

**Trace** records observable steps only (`ok`/`warning`/`failed`/`skipped`). It never
contains model reasoning or chain-of-thought.

**422** for a malformed request: empty query, zero or more than two `asset_ids`, or
duplicate ids.

Issue codes seen in `issues[]`: `unknown_input`, `no_inputs`, `unsupported_query`,
`tool_unavailable`, `incompatible_inputs`, `too_few_images`, `too_many_images`,
`modality_mismatch`, `modality_unsupported`, `parameter_not_allowed`,
`parameter_out_of_range`, `parameter_type_invalid`, `tool_not_registered`, `execution_failed`.

### `GET /api/v1/analysis/{id}`
The stored analysis. **404** if unknown. (The stored record omits `plan`; use the response
from `POST` for full plan fidelity.)

### `GET /api/v1/analysis/{id}/trace`
`{ id, status, task, trace[] }`.

### `GET /api/v1/analysis/{id}/evidence`
`{ id, status, task, evidence[] }`.

### `GET /api/v1/analysis/{id}/report`
Returns a **self-contained HTML report** for the analysis as a file attachment
(`Content-Disposition: attachment; filename="satquery-report-<id>.html"`). The report is built
entirely from the persisted record — query, task and tool/version, answer, confidence and its
signals, evidence (images embedded as base64 data URIs so it works offline), the observable
trace, provenance, and limitations. Every dynamic value is HTML-escaped, so a hostile filename
or query cannot inject markup. It contains no chain-of-thought. Any browser can print it to PDF.
**404** if the analysis is unknown.

### `GET /api/v1/history`
`{ count, items[] }` — recent analyses (`id`, `created_at`, `status`, `task`, `query`,
`confidence_level`), newest first. Query: `limit` (1–200, default 50).

## Query safety
Queries are untrusted input. They are unicode-normalized, stripped of control characters,
length-bounded, and scanned for instruction-like fragments (e.g. "ignore previous
instructions", "reveal your system prompt"). Detected fragments are removed and reported as
a `warning` entry in the trace; the imagery question is still answered. Available tools and
permitted parameters come only from the registry, never from the query.
