"""Unit tests for raster metadata extraction, modality hints, and windowed reads."""

from __future__ import annotations

from geospatial import (
    infer_modality,
    is_readable_raster,
    iter_tiles,
    read_decimated,
    read_metadata,
    read_window,
    render_preview,
)

from .raster_fixtures import write_corrupt_tiff, write_geotiff, write_plain_png


def test_metadata_from_georeferenced_geotiff(tmp_path):
    path = write_geotiff(tmp_path / "opt.tif", width=128, height=64, bands=3)
    meta = read_metadata(path)

    assert (meta.width, meta.height) == (128, 64)
    assert meta.band_count == 3
    assert meta.dtype == "uint8"
    assert meta.driver == "GTiff"
    assert meta.is_georeferenced is True
    assert meta.crs_epsg == 4326
    assert meta.resolution is not None
    assert round(meta.resolution[0], 6) == 0.001
    assert meta.bounds is not None and meta.bounds[0] == 77.0
    assert len(meta.bands) == 3
    assert meta.bands[0].index == 1


def test_metadata_preserves_nodata_and_tags(tmp_path):
    path = write_geotiff(
        tmp_path / "sar.tif",
        bands=1,
        dtype="float32",
        nodata=-9999.0,
        tags={"SENSOR": "Sentinel-1"},
    )
    meta = read_metadata(path)

    assert meta.nodata == -9999.0
    assert meta.bands[0].nodata == -9999.0
    assert any("sentinel-1" in v.lower() for v in meta.tags.values())


def test_plain_png_is_not_georeferenced(tmp_path):
    """Georeferencing must be reported honestly rather than fabricated."""
    path = write_plain_png(tmp_path / "bench.png")
    meta = read_metadata(path)

    assert meta.is_georeferenced is False
    assert meta.crs is None
    assert meta.bounds is None
    assert meta.resolution is None


def test_modality_inference(tmp_path):
    optical = read_metadata(write_geotiff(tmp_path / "o.tif", bands=3, dtype="uint8"))
    sar = read_metadata(write_geotiff(tmp_path / "s.tif", bands=1, dtype="float32"))
    tagged = read_metadata(
        write_geotiff(tmp_path / "t.tif", bands=3, tags={"SENSOR": "Sentinel-1 SAR"})
    )

    assert str(infer_modality(optical)) == "optical"
    assert str(infer_modality(sar)) == "sar"
    # An explicit SAR sensor tag outweighs the band-count heuristic.
    assert str(infer_modality(tagged)) == "sar"


def test_read_window_is_bounded_and_clipped(tmp_path):
    path = write_geotiff(tmp_path / "w.tif", width=100, height=80, bands=2)

    block = read_window(path, row_off=10, col_off=20, height=16, width=32)
    assert block.shape == (2, 16, 32)

    # Requests beyond the extent are clipped rather than raising.
    clipped = read_window(path, row_off=70, col_off=90, height=50, width=50)
    assert clipped.shape == (2, 10, 10)


def test_read_decimated_downsamples(tmp_path):
    path = write_geotiff(tmp_path / "big.tif", width=800, height=400, bands=3)
    data, scale = read_decimated(path, max_side=200)

    assert max(data.shape[1], data.shape[2]) <= 200
    assert scale < 1.0


def test_iter_tiles_covers_every_pixel(tmp_path):
    path = write_geotiff(tmp_path / "tiled.tif", width=130, height=70, bands=1)

    covered = 0
    for window, array in iter_tiles(path, tile_size=64):
        assert array.shape[1] == window.height
        assert array.shape[2] == window.width
        covered += window.height * window.width

    assert covered == 130 * 70


def test_render_preview_writes_png(tmp_path):
    source = write_geotiff(tmp_path / "src.tif", width=300, height=150, bands=3)
    destination = tmp_path / "out" / "preview.png"

    width, height = render_preview(source, destination, max_side=100)

    assert destination.is_file()
    assert max(width, height) <= 100


def test_render_preview_handles_single_band_sar(tmp_path):
    source = write_geotiff(tmp_path / "sar.tif", bands=1, dtype="float32")
    destination = tmp_path / "sar.png"

    render_preview(source, destination)
    assert destination.is_file()


def test_is_readable_raster_rejects_corrupt_file(tmp_path):
    assert is_readable_raster(write_geotiff(tmp_path / "ok.tif")) is True
    assert is_readable_raster(write_corrupt_tiff(tmp_path / "bad.tif")) is False
