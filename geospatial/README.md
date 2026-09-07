# geospatial/ — Remote-sensing raster IO & processing

Reusable geospatial utilities shared by the backend and ML layers. Populated in
**Phase 2 (raster pipeline)**.

Planned modules:

```text
io.py            Windowed/tiled GeoTIFF reading (never load huge rasters fully into RAM)
metadata.py      CRS, transform, bounds, resolution, bands, dtype extraction
validation.py    Format/MIME/corruption + remote-sensing metadata validation
alignment.py     Pair overlap, CRS compatibility, resampling, co-registration checks
tiling.py        Tiling / chunking for large-raster processing
render.py        Safe preview rendering (band selection, normalization) for the UI
```

Rules (see `.kiro/steering/geospatial-rules.md`): correct CRS/transform handling,
windowed reads, explicit resampling, and no silent analysis of incompatible pairs.
