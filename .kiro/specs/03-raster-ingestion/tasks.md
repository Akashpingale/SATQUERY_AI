# Spec 03 — Raster Ingestion · Tasks

- [x] 1. Geospatial dependencies
  - rasterio 1.4.4 (bundled GDAL 3.10.3), numpy, pyproj, shapely, Pillow, python-multipart
  - verified on Windows via wheels (no system GDAL build required)
  - recorded in `backend/pyproject.toml` and root `pyproject.toml`
  - _Reqs: all_

- [x] 2. `geospatial/` library
  - [x] 2.1 `types.py` — metadata/validation/compatibility dataclasses
  - [x] 2.2 `metadata.py` — header-only `read_metadata`, `infer_modality` hint
  - [x] 2.3 `io.py` — `read_window`, `iter_tiles`, `read_decimated`, `is_readable_raster`
  - [x] 2.4 `validation.py` — extension + real-open + sanity limits
  - [x] 2.5 `alignment.py` — `check_pair_compatibility` with pyproj reprojection
  - [x] 2.6 `render.py` — downsampled PNG preview (percentile stretch, RGB/grayscale)
  - _Reqs: metadata, windowed reads, pair validation_

- [x] 3. Upload security
  - filename sanitization (traversal/unicode/null/unsafe chars/unknown extensions)
  - collision-free stored names; `resolve_within` path confinement
  - streaming size limit with partial-file cleanup; content-type allowlist
  - _Reqs: untrusted input_

- [x] 4. Storage + persistence
  - storage root with `uploads/processed/evidence/reports/temp`
  - `RasterAsset` model + Alembic `0002_raster_assets`
  - _Reqs: metadata persistence_

- [x] 5. Uploads API
  - `POST /uploads`, `GET /uploads`, `GET /uploads/{id}`, `GET /uploads/{id}/preview`,
    `DELETE /uploads/{id}`, `POST /uploads/compatibility`
  - _Reqs: ingestion, previews, pair validation_

- [x] 6. Registry
  - `raster_preprocessor` and `registration_validator` → `available` v1.0.0
  - _Reqs: honest capability status_

- [x] 7. Backend tests (53 new, 60 total)
  - metadata/IO/preview units; validation + security negatives; alignment; API integration
  - real GeoTIFF/PNG fixtures written by Rasterio (not mocks)
  - _Reqs: all_

- [x] 8. Frontend ingestion UX
  - drag-drop `UploadSlot` ×4, `MetadataPanel`, `IssueList`, `CompatibilityStatus`,
    `ImageViewer` (zoom/pan/fit), `useUploads`
  - _Reqs: UI surface for ingestion_

- [x] 9. Frontend tests (20 new, 23 total) + strict build
  - _Reqs: all_

- [x] 10. Documentation
  - spec design/tasks, `docs/traceability.md`, `docs/architecture.md`, `docs/api.md`
  - _Reqs: documentation accuracy_

## Deferred to later phases (intentional)
- Reprojection/resampling **execution** (checks exist now; the actual resampling runs in
  Phase 6/7 where change detection and fusion consume aligned pairs).
- Tile-serving endpoint for very large rasters in the browser (previews suffice today).
- OpenCV / scikit-image / GeoPandas — not needed yet; added when a feature requires them.
- Preprocessing cache (previews are generated once at ingest; a cache layer arrives with
  Phase 4 inference).
