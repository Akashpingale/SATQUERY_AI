"""Pair-compatibility tests for temporal and optical-SAR inputs."""

from __future__ import annotations

from geospatial import check_pair_compatibility, read_metadata

from .raster_fixtures import write_geotiff, write_plain_png


def _meta(path):
    return read_metadata(path)


def test_identical_grid_is_compatible(tmp_path):
    a = _meta(write_geotiff(tmp_path / "a.tif"))
    b = _meta(write_geotiff(tmp_path / "b.tif", bands=1, dtype="float32"))

    result = check_pair_compatibility(a, b)

    assert result.compatible is True
    assert result.overlap_ratio == 1.0
    assert result.same_crs is True
    assert result.resampling_required is False
    assert result.alignment_required is False


def test_non_overlapping_pair_is_rejected(tmp_path):
    a = _meta(write_geotiff(tmp_path / "a.tif", origin=(77.0, 13.0)))
    b = _meta(write_geotiff(tmp_path / "b.tif", origin=(10.0, 50.0)))

    result = check_pair_compatibility(a, b)

    assert result.compatible is False
    assert "no_overlap" in [i.code for i in result.issues]


def test_partial_overlap_warns_and_requires_alignment(tmp_path):
    # Shift by half the 64 px * 0.001 deg footprint.
    a = _meta(write_geotiff(tmp_path / "a.tif", width=64, height=64, origin=(77.0, 13.0)))
    b = _meta(write_geotiff(tmp_path / "b.tif", width=64, height=64, origin=(77.032, 13.0)))

    result = check_pair_compatibility(a, b)

    codes = [i.code for i in result.issues]
    assert "partial_overlap" in codes
    assert result.alignment_required is True
    assert result.overlap_ratio is not None and 0.3 < result.overlap_ratio < 1.0


def test_resolution_mismatch_requires_resampling(tmp_path):
    a = _meta(write_geotiff(tmp_path / "a.tif", width=64, height=64, resolution=0.001))
    b = _meta(write_geotiff(tmp_path / "b.tif", width=32, height=32, resolution=0.002))

    result = check_pair_compatibility(a, b)

    assert result.resampling_required is True
    assert "resolution_mismatch" in [i.code for i in result.issues]


def test_different_crs_is_reprojected_before_overlap_check(tmp_path):
    """A UTM/WGS84 pair over the same ground area must still be found overlapping."""
    # EPSG:32643 (UTM 43N) around 77.0E, 13.0N -> easting ~717000, northing ~1438000.
    wgs = _meta(write_geotiff(tmp_path / "wgs.tif", width=64, height=64, origin=(77.0, 13.0)))
    utm = _meta(
        write_geotiff(
            tmp_path / "utm.tif",
            width=64,
            height=64,
            crs="EPSG:32643",
            origin=(717000.0, 1438000.0),
            resolution=100.0,
        )
    )

    result = check_pair_compatibility(wgs, utm)

    assert result.same_crs is False
    assert "crs_mismatch" in [i.code for i in result.issues]
    # Reprojection must have produced a real overlap measurement.
    assert result.overlap_ratio is not None
    assert result.overlap_ratio > 0.0
    assert result.alignment_required is True


def test_non_georeferenced_same_size_pair_warns_but_allows(tmp_path):
    a = _meta(write_plain_png(tmp_path / "a.png", 32, 32))
    b = _meta(write_plain_png(tmp_path / "b.png", 32, 32))

    result = check_pair_compatibility(a, b)

    codes = [i.code for i in result.issues]
    assert "not_georeferenced" in codes
    assert result.compatible is True  # warning only
    assert result.alignment_required is True
    assert result.same_crs is None


def test_non_georeferenced_different_size_pair_is_rejected(tmp_path):
    a = _meta(write_plain_png(tmp_path / "a.png", 32, 32))
    b = _meta(write_plain_png(tmp_path / "b.png", 48, 20))

    result = check_pair_compatibility(a, b)

    assert result.compatible is False
    assert "dimension_mismatch" in [i.code for i in result.issues]
