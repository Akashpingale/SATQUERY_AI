"""Phase 6 unit tests: co-registration, band roles, and the change detector itself.

These tests assert **measured behaviour against known ground truth**, not that functions
return without raising. A synthetic pair whose changed block is exactly 30x40 pixels must be
measured as 1200 changed pixels; an unchanged pair must measure zero.

Everything here is pure NumPy/SciPy/Rasterio, so it runs anywhere - no GPU, no model weights,
no network.
"""

from __future__ import annotations

import numpy as np
import pytest
from geospatial import (
    DEFAULT_ALIGN_MAX_SIDE,
    AlignmentError,
    align_pair,
    read_metadata,
    resolve_band_roles,
)
from geospatial.bands import normalized_difference
from ml.inference.change_map import crop_pair, render_change_map, render_comparison_strip
from ml.inference.temporal import (
    ALGORITHM_VERSION,
    ChangeConfig,
    ChangeMethod,
    detect_change,
    otsu_threshold,
    radiometric_normalize,
    robust_spread,
)
from PIL import Image

from .raster_fixtures import textured_scene, with_block, write_geotiff

# A projected, metric CRS so ground areas in square metres are meaningful.
UTM = "EPSG:32643"
UTM_ORIGIN = (500_000.0, 1_500_000.0)
RES_M = 10.0


def _pair(tmp_path, first, second, **kwargs):
    """Write two rasters onto the same metric grid and align them."""
    options = {
        "crs": UTM,
        "origin": UTM_ORIGIN,
        "resolution": RES_M,
        "dtype": "uint16",
        **kwargs,
    }
    a = write_geotiff(tmp_path / "t1.tif", array=first, **options)
    b = write_geotiff(tmp_path / "t2.tif", array=second, **options)
    return align_pair(a, b)


# --- threshold and statistics primitives ------------------------------------------


def test_otsu_finds_the_split_between_two_separated_modes():
    values = np.concatenate([np.full(500, 0.1), np.full(500, 0.9)])
    threshold = otsu_threshold(values)
    assert 0.1 < threshold < 0.9


def test_otsu_handles_empty_and_constant_input():
    assert otsu_threshold(np.array([])) == 0.0
    assert otsu_threshold(np.full(10, 0.5)) == pytest.approx(0.5)


def test_robust_spread_ignores_an_outlier_minority():
    """A MAD-based sigma must not be dragged by real change, which is the whole point."""
    clean = np.concatenate([np.zeros(900), np.full(100, 1000.0)])
    median, sigma = robust_spread(clean)
    assert median == pytest.approx(0.0)
    # A standard deviation here would be ~300; MAD-scaled sigma stays at zero.
    assert sigma == pytest.approx(0.0)
    assert float(np.std(clean)) > 100.0


def test_radiometric_normalize_removes_a_global_gain_and_offset():
    scene = textured_scene(size=48)
    shifted = scene * 1.4 + 250.0
    valid = np.ones(scene.shape[1:], dtype=bool)

    normalised, parameters = radiometric_normalize(scene, shifted, valid)

    assert len(parameters) == scene.shape[0]
    # Recovering the original means the illumination difference is gone.
    assert np.allclose(normalised, scene, rtol=0.02, atol=2.0)


def test_radiometric_normalize_leaves_a_constant_band_alone():
    """A flat band has no contrast to match; dividing by ~0 would manufacture change."""
    reference = np.zeros((1, 8, 8), dtype="float32")
    target = np.full((1, 8, 8), 5.0, dtype="float32")
    valid = np.ones((8, 8), dtype=bool)

    _normalised, parameters = radiometric_normalize(reference, target, valid)

    assert parameters[0]["gain"] == pytest.approx(1.0)


def test_normalized_difference_guards_against_divide_by_zero():
    first = np.array([[1.0, 0.0]], dtype="float32")
    second = np.array([[1.0, 0.0]], dtype="float32")
    result = normalized_difference(first, second)
    assert np.all(np.isfinite(result))
    assert result[0, 1] == 0.0


# --- co-registration --------------------------------------------------------------


def test_align_pair_puts_a_matching_pair_on_a_shared_grid(tmp_path):
    scene = textured_scene(size=64)
    aligned = _pair(tmp_path, scene, scene)

    assert aligned.width == 64
    assert aligned.height == 64
    assert aligned.band_count == 3
    assert aligned.georeferenced is True
    assert aligned.crs == UTM
    assert aligned.pixel_area == pytest.approx(RES_M * RES_M)
    assert aligned.valid_fraction == pytest.approx(1.0)
    assert any("resampled both images" in note for note in aligned.operations)


def test_align_pair_reprojects_a_crs_mismatch_and_records_it(tmp_path):
    """A CRS mismatch is resolved by real reprojection, and the operation is reported."""
    scene = textured_scene(size=64)
    a = write_geotiff(
        tmp_path / "utm.tif", array=scene, crs=UTM, origin=UTM_ORIGIN, resolution=RES_M,
        dtype="uint16",
    )
    # The same ground area expressed in WGS84 degrees.
    b = write_geotiff(
        tmp_path / "wgs.tif", array=scene, crs="EPSG:4326", origin=(75.0, 13.57),
        resolution=0.0001, dtype="uint16",
    )

    aligned = align_pair(a, b)

    assert aligned.crs == UTM
    assert any("reprojected" in note for note in aligned.operations)
    assert aligned.width >= 2 and aligned.height >= 2


def test_align_pair_crops_to_the_overlap_of_a_partial_pair(tmp_path):
    scene = textured_scene(size=64)
    a = write_geotiff(
        tmp_path / "a.tif", array=scene, crs=UTM, origin=UTM_ORIGIN, resolution=RES_M,
        dtype="uint16",
    )
    # Shift east/south by 32 px so only a quadrant overlaps.
    shifted_origin = (UTM_ORIGIN[0] + 32 * RES_M, UTM_ORIGIN[1] - 32 * RES_M)
    b = write_geotiff(
        tmp_path / "b.tif", array=scene, crs=UTM, origin=shifted_origin, resolution=RES_M,
        dtype="uint16",
    )

    aligned = align_pair(a, b)

    assert aligned.width == 32
    assert aligned.height == 32
    assert any("cropped to the shared area" in note for note in aligned.operations)


def test_align_pair_uses_the_coarser_resolution_so_nothing_is_upsampled(tmp_path):
    scene = textured_scene(size=64)
    a = write_geotiff(
        tmp_path / "fine.tif", array=scene, crs=UTM, origin=UTM_ORIGIN, resolution=RES_M,
        dtype="uint16",
    )
    b = write_geotiff(
        tmp_path / "coarse.tif", array=scene, crs=UTM, origin=UTM_ORIGIN,
        resolution=RES_M * 2, dtype="uint16",
    )

    aligned = align_pair(a, b)

    assert aligned.pixel_area == pytest.approx((RES_M * 2) ** 2)
    assert any("coarser pixel size" in note for note in aligned.operations)


def test_align_pair_rejects_a_non_overlapping_pair(tmp_path):
    scene = textured_scene(size=48)
    a = write_geotiff(
        tmp_path / "a.tif", array=scene, crs=UTM, origin=UTM_ORIGIN, resolution=RES_M,
        dtype="uint16",
    )
    b = write_geotiff(
        tmp_path / "b.tif", array=scene, crs=UTM, origin=(900_000.0, 1_900_000.0),
        resolution=RES_M, dtype="uint16",
    )

    with pytest.raises(AlignmentError) as excinfo:
        align_pair(a, b)
    assert excinfo.value.code == "no_overlap"


def test_align_pair_rejects_non_georeferenced_images_of_different_sizes(tmp_path):
    a = write_geotiff(tmp_path / "a.tif", width=64, height=64, crs=None)
    b = write_geotiff(tmp_path / "b.tif", width=48, height=48, crs=None)

    with pytest.raises(AlignmentError) as excinfo:
        align_pair(a, b)
    assert excinfo.value.code == "dimension_mismatch"


def test_align_pair_accepts_non_georeferenced_images_of_equal_size(tmp_path):
    """Equal pixel grids are the one case where an unreferenced pair can be compared."""
    a = write_geotiff(tmp_path / "a.tif", width=64, height=64, crs=None, seed=1)
    b = write_geotiff(tmp_path / "b.tif", width=64, height=64, crs=None, seed=2)

    aligned = align_pair(a, b)

    assert aligned.georeferenced is False
    assert aligned.crs is None
    assert aligned.pixel_area is None
    assert any("pre-registered" in note for note in aligned.operations)


def test_align_pair_truncates_to_the_common_band_count(tmp_path):
    a = write_geotiff(tmp_path / "a.tif", bands=3, crs=UTM, origin=UTM_ORIGIN, resolution=RES_M)
    b = write_geotiff(tmp_path / "b.tif", bands=1, crs=UTM, origin=UTM_ORIGIN, resolution=RES_M)

    aligned = align_pair(a, b)

    assert aligned.band_count == 1
    assert any("compared the first 1 band" in note for note in aligned.operations)


def test_align_pair_bounds_memory_by_decimating_large_inputs(tmp_path):
    """A large pair must be decimated to the configured cap, not loaded at full size."""
    big = textured_scene(size=900)
    aligned = _pair(tmp_path, big, big)

    assert max(aligned.width, aligned.height) <= DEFAULT_ALIGN_MAX_SIDE
    assert any("decimated" in note for note in aligned.operations)


# --- band roles: declared only, never guessed --------------------------------------


def test_band_roles_are_resolved_from_declared_descriptions(tmp_path):
    path = write_geotiff(
        tmp_path / "named.tif", bands=4, band_names=["Blue", "Green", "Red", "NIR"]
    )
    roles = resolve_band_roles(read_metadata(path))

    assert (roles.blue, roles.green, roles.red, roles.nir) == (1, 2, 3, 4)
    assert roles.ndvi_available is True
    assert roles.ndwi_available is True
    assert roles.source == "band descriptions"


def test_band_roles_recognise_sentinel2_band_codes(tmp_path):
    path = write_geotiff(tmp_path / "s2.tif", bands=4, band_names=["B02", "B03", "B04", "B08"])
    roles = resolve_band_roles(read_metadata(path))
    assert (roles.blue, roles.green, roles.red, roles.nir) == (1, 2, 3, 4)


def test_band_roles_stay_unknown_when_nothing_is_declared(tmp_path):
    """Band order is not standardised, so an undeclared raster must yield no roles."""
    path = write_geotiff(tmp_path / "plain.tif", bands=4)
    roles = resolve_band_roles(read_metadata(path))

    assert roles.red is None and roles.nir is None
    assert roles.ndvi_available is False
    assert roles.source == "undeclared"


# --- the detector: known ground truth ---------------------------------------------


def test_identical_pair_measures_no_change(tmp_path):
    scene = textured_scene(size=64)
    result = detect_change(_pair(tmp_path, scene, scene))

    assert result.inconclusive is False
    assert result.changed_pixels == 0
    assert result.changed_fraction == 0.0
    assert result.region_count == 0
    assert result.significant is False
    assert result.registration.peak_offset == (0, 0)
    assert result.registration.peak_correlation == pytest.approx(1.0, abs=1e-3)


def test_illumination_only_difference_is_not_reported_as_change(tmp_path):
    """A brighter, higher-contrast second date is normalised away, not called change."""
    scene = textured_scene(size=64)
    brighter = np.clip(scene * 1.25 + 300.0, 1, 65535)

    result = detect_change(_pair(tmp_path, scene, brighter))

    assert result.significant is False
    assert result.changed_fraction < 0.005


def test_independently_noised_pair_is_not_reported_as_change(tmp_path):
    """Sensor noise must not survive coherence filtering."""
    scene = textured_scene(size=96, seed=3)
    rng = np.random.default_rng(99)
    renoised = np.clip(scene + rng.normal(0, 45, scene.shape), 1, 65535)

    result = detect_change(_pair(tmp_path, scene, renoised))

    assert result.significant is False
    assert result.region_count == 0
    # The raw threshold pass may flag scattered pixels; coherence filtering removes them.
    assert result.changed_pixels == 0


def test_strong_local_change_is_measured_at_its_true_size(tmp_path):
    scene = textured_scene(size=96, seed=11)
    changed = with_block(scene, rows=(20, 50), cols=(30, 70), value=5200.0)
    expected_pixels = (50 - 20) * (70 - 30)

    result = detect_change(_pair(tmp_path, scene, changed))

    assert result.significant is True
    assert result.changed_pixels == expected_pixels
    assert result.region_count == 1
    region = result.regions[0]
    assert region.pixel_count == expected_pixels
    # Box is (x_min, y_min, x_max, y_max) inclusive on the common grid.
    assert region.box == (30, 20, 69, 49)
    assert region.brightness_delta > 0
    assert region.area_m2 == pytest.approx(expected_pixels * RES_M * RES_M)
    assert region.centroid_lonlat is not None
    assert result.method is ChangeMethod.CHANGE_VECTOR
    assert result.algorithm_version == ALGORITHM_VERSION


def test_weak_local_change_is_still_detected(tmp_path):
    scene = textured_scene(size=96, seed=13)
    changed = with_block(scene, rows=(60, 85), cols=(50, 78), scale=1.18)
    expected_pixels = (85 - 60) * (78 - 50)

    result = detect_change(_pair(tmp_path, scene, changed))

    assert result.significant is True
    assert result.changed_pixels == expected_pixels


def test_a_darker_change_is_detected_and_reported_as_darker(tmp_path):
    scene = textured_scene(size=96, seed=17)
    changed = with_block(scene, rows=(40, 70), cols=(20, 55), value=200.0)

    result = detect_change(_pair(tmp_path, scene, changed))

    assert result.significant is True
    assert result.brightness_delta < 0
    assert result.regions[0].brightness_delta < 0


def test_two_separate_changes_are_reported_as_two_regions(tmp_path):
    scene = textured_scene(size=120, seed=19)
    changed = with_block(scene, rows=(10, 40), cols=(10, 45), value=5000.0)
    changed = with_block(changed, rows=(80, 105), cols=(70, 100), value=150.0)

    result = detect_change(_pair(tmp_path, scene, changed))

    assert result.region_count == 2
    assert len(result.regions) == 2
    # Largest first, so region 1 is the bigger block.
    assert result.regions[0].pixel_count >= result.regions[1].pixel_count
    assert result.regions[0].pixel_count == (40 - 10) * (45 - 10)


def test_a_dominant_change_does_not_hide_a_weaker_one(tmp_path):
    """Regression: a global Otsu split let a strong change mask a genuine weaker change."""
    scene = textured_scene(size=140, seed=23)
    changed = with_block(scene, rows=(10, 50), cols=(10, 60), value=6000.0)  # dominant
    changed = with_block(changed, rows=(95, 120), cols=(90, 125), value=600.0)  # weaker

    result = detect_change(_pair(tmp_path, scene, changed))

    assert result.region_count == 2, "the weaker change must survive thresholding"
    sizes = sorted(region.pixel_count for region in result.regions)
    assert sizes == [(120 - 95) * (125 - 90), (50 - 10) * (60 - 10)]


def test_misregistered_pair_is_reported_inconclusive_not_as_change(tmp_path):
    """Comparing displaced pixels cannot separate shift artifacts from real change."""
    scene = textured_scene(size=96, seed=29)
    shifted = np.roll(scene, shift=2, axis=2)

    result = detect_change(_pair(tmp_path, scene, shifted))

    assert result.inconclusive is True
    assert "misregistered" in result.inconclusive_reason.lower()
    assert result.changed_pixels == 0
    assert result.registration.shift_pixels == 2
    assert result.registration.shift_explains > 0.5


def test_a_strong_change_is_not_mistaken_for_misregistration(tmp_path):
    """The shift-explains discriminator must not fire on a large real change."""
    scene = textured_scene(size=96, seed=31)
    changed = with_block(scene, rows=(20, 60), cols=(20, 60), value=5400.0)

    result = detect_change(_pair(tmp_path, scene, changed))

    assert result.inconclusive is False
    assert result.registration.shift_explains < 0.5
    assert result.significant is True


def test_sar_pair_uses_the_log_ratio_formulation(tmp_path):
    scene = textured_scene(size=64, bands=1, seed=37)
    changed = with_block(scene, rows=(15, 40), cols=(15, 45), scale=3.0)

    result = detect_change(_pair(tmp_path, scene, changed), is_sar=True)

    assert result.method is ChangeMethod.LOG_RATIO
    assert result.significant is True


def test_log_ratio_falls_back_when_values_are_not_strictly_positive(tmp_path):
    """A ratio needs positive data; zeros force the change-vector formulation instead."""
    scene = textured_scene(size=64, bands=1, seed=41)
    scene[:, 0:5, 0:5] = 0.0
    changed = with_block(scene, rows=(20, 45), cols=(20, 50), scale=2.5)

    result = detect_change(_pair(tmp_path, scene, changed), is_sar=True)

    assert result.method is ChangeMethod.CHANGE_VECTOR


def test_thresholds_and_algorithm_version_are_reported(tmp_path):
    scene = textured_scene(size=64, seed=43)
    changed = with_block(scene, rows=(10, 35), cols=(10, 40), value=4800.0)

    stats = detect_change(_pair(tmp_path, scene, changed)).stats_dict()

    assert stats["algorithm_version"] == ALGORITHM_VERSION
    assert stats["threshold_method"]
    assert stats["changed_pixels"] > 0
    assert stats["registration_peak_correlation"] <= 1.0
    assert "otsu_threshold" in stats


def test_min_region_pixels_suppresses_specks(tmp_path):
    scene = textured_scene(size=96, seed=47)
    changed = with_block(scene, rows=(40, 43), cols=(40, 43), value=6000.0)  # 9 px

    strict = detect_change(_pair(tmp_path, scene, changed), config=ChangeConfig())
    permissive = detect_change(
        _pair(tmp_path, scene, changed), config=ChangeConfig(min_region_pixels=4)
    )

    assert strict.region_count == 0
    assert permissive.region_count == 1


def test_area_in_square_metres_is_withheld_for_a_geographic_crs(tmp_path):
    """Degrees squared is not an area, so no metric figure may be reported."""
    scene = textured_scene(size=64, seed=53)
    changed = with_block(scene, rows=(10, 40), cols=(10, 45), value=5000.0)

    result = detect_change(
        _pair(tmp_path, scene, changed, crs="EPSG:4326", origin=(77.0, 13.0), resolution=0.0001)
    )

    assert result.significant is True
    assert result.changed_area_m2 is None
    assert result.area_units == "pixels"
    assert all(region.area_m2 is None for region in result.regions)
    assert any("square metres is not reported" in warning for warning in result.warnings)


def test_no_valid_overlap_is_inconclusive(tmp_path):
    """All-nodata inputs share no comparable pixels, so nothing may be claimed."""
    blank = np.zeros((3, 64, 64), dtype="float32")
    aligned = _pair(tmp_path, blank, blank, nodata=0)

    result = detect_change(aligned)

    assert result.inconclusive is True
    assert result.changed_pixels == 0


# --- spectral indices -------------------------------------------------------------


def _ndvi_pair(tmp_path, *, declare_bands: bool):
    size = 96
    rng = np.random.default_rng(59)
    red = np.clip(700 + rng.normal(0, 25, (size, size)), 1, 65535)
    green = np.clip(950 + rng.normal(0, 25, (size, size)), 1, 65535)
    nir = np.clip(3100 + rng.normal(0, 55, (size, size)), 1, 65535)
    before = np.stack([red, green, nir])
    after = before.copy()
    after[2, 30:70, 30:70] = 900.0  # NIR collapses: vegetation loss
    after[0, 30:70, 30:70] = 1500.0  # red rises

    names = ["Red", "Green", "NIR"] if declare_bands else None
    options = {
        "crs": UTM, "origin": UTM_ORIGIN, "resolution": RES_M, "dtype": "uint16",
        "band_names": names,
    }
    a = write_geotiff(tmp_path / "v1.tif", array=before, **options)
    b = write_geotiff(tmp_path / "v2.tif", array=after, **options)
    roles_a = resolve_band_roles(read_metadata(a))
    roles_b = resolve_band_roles(read_metadata(b))
    return align_pair(a, b), roles_a, roles_b


def test_ndvi_is_measured_when_bands_are_declared(tmp_path):
    aligned, roles_a, roles_b = _ndvi_pair(tmp_path, declare_bands=True)
    result = detect_change(aligned, roles_first=roles_a, roles_second=roles_b)

    ndvi = result.index_by_name("NDVI")
    assert ndvi is not None and ndvi.available is True
    assert ndvi.mean_before > ndvi.mean_after, "NDVI must fall when NIR collapses"
    assert ndvi.delta < 0
    assert ndvi.delta_in_changed is not None
    assert ndvi.delta_in_changed < ndvi.delta, "the loss is concentrated in the changed area"

    ndwi = result.index_by_name("NDWI")
    assert ndwi is not None and ndwi.available is True


def test_indices_are_refused_when_bands_are_undeclared(tmp_path):
    """Guessing band order would produce a plausible, unfounded index."""
    aligned, roles_a, roles_b = _ndvi_pair(tmp_path, declare_bands=False)
    result = detect_change(aligned, roles_first=roles_a, roles_second=roles_b)

    for name in ("NDVI", "NDWI"):
        index = result.index_by_name(name)
        assert index is not None
        assert index.available is False
        assert index.reason
        assert index.delta is None


def test_indices_are_refused_when_no_roles_are_supplied(tmp_path):
    scene = textured_scene(size=64, seed=61)
    result = detect_change(_pair(tmp_path, scene, scene))

    assert [item.available for item in result.indices] == [False, False]
    assert all("guessing band order" in item.reason for item in result.indices)


# --- rendered artifacts -----------------------------------------------------------


def test_change_map_is_rendered_and_marks_only_flagged_pixels(tmp_path):
    scene = textured_scene(size=96, seed=67)
    changed = with_block(scene, rows=(20, 50), cols=(25, 60), value=5600.0)
    aligned = _pair(tmp_path, scene, changed)
    result = detect_change(aligned)

    destination = tmp_path / "change.png"
    width, height = render_change_map(aligned, result, destination)

    assert destination.is_file()
    assert (width, height) == (aligned.width, aligned.height)

    pixels = np.asarray(Image.open(destination).convert("RGB")).astype("int16")
    # A tinted pixel is warm (red channel clearly above blue); the grey backdrop is not.
    warm = (pixels[:, :, 0] - pixels[:, :, 2]) > 40
    assert warm.sum() > 0, "flagged areas must be visibly tinted"
    # Every tinted pixel must sit inside a reported region's box, i.e. be backed by a
    # measurement rather than decoration. Outlines are drawn on the box edge, so the boxes
    # are the correct containment test.
    inside = np.zeros(warm.shape, dtype=bool)
    for region in result.regions:
        x0, y0, x1, y1 = region.box
        inside[y0 : y1 + 1, x0 : x1 + 1] = True
    assert not np.any(warm & ~inside), "no tint may appear outside a measured region"


def test_unchanged_pair_renders_a_change_map_with_no_tint(tmp_path):
    scene = textured_scene(size=64, seed=71)
    aligned = _pair(tmp_path, scene, scene)
    result = detect_change(aligned)

    destination = tmp_path / "flat.png"
    render_change_map(aligned, result, destination)

    pixels = np.asarray(Image.open(destination).convert("RGB")).astype("int16")
    warm = (pixels[:, :, 0] - pixels[:, :, 2]) > 40
    assert warm.sum() == 0, "an unchanged pair must not be painted as changed"


def test_comparison_strip_has_three_panels_on_the_shared_grid(tmp_path):
    scene = textured_scene(size=80, seed=73)
    changed = with_block(scene, rows=(20, 45), cols=(20, 50), value=5200.0)
    aligned = _pair(tmp_path, scene, changed)
    result = detect_change(aligned)

    destination = tmp_path / "strip.png"
    width, height = render_comparison_strip(aligned, result, destination)

    assert destination.is_file()
    assert width >= aligned.width * 3
    assert height > aligned.height


def test_crop_pair_returns_both_dates_for_a_usable_region(tmp_path):
    scene = textured_scene(size=96, seed=79)
    changed = with_block(scene, rows=(20, 60), cols=(20, 60), value=5000.0)
    aligned = _pair(tmp_path, scene, changed)
    result = detect_change(aligned)

    crops = crop_pair(aligned, result.regions[0].box)

    assert crops is not None
    before, after = crops
    assert before.size == after.size
    assert before.mode == "RGB"
    assert np.asarray(before).shape[2] == 3


def test_crop_pair_declines_a_region_too_small_to_read(tmp_path):
    """A handful of pixels cannot support a land-cover reading."""
    scene = textured_scene(size=64, seed=83)
    aligned = _pair(tmp_path, scene, scene)

    assert crop_pair(aligned, (10, 10, 14, 14)) is None