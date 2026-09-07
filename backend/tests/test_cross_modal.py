"""Phase 7 unit tests: modality alignment, SAR/optical feature extraction, and the
cross-modal complementarity measurement.

Assertions target measured behaviour against a synthetic optical/SAR pair with known features
(a water strip both modalities should agree on, a structured block only SAR should reveal).
Pure NumPy/SciPy/Rasterio: no weights, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from geospatial import (
    AlignmentError,
    align_modalities,
    read_metadata,
    resolve_band_roles,
)
from ml.inference.fusion import (
    FUSION_ALGORITHM_VERSION,
    CrossModalQuestion,
    analyze_cross_modal,
    classify_cross_modal_question,
)
from ml.inference.fusion_render import (
    optical_pil,
    render_aligned_sar,
    render_complementary_map,
    render_fusion_composite,
)
from PIL import Image

from .raster_fixtures import optical_sar_scene, textured_scene, write_geotiff

UTM = "EPSG:32643"
UTM_ORIGIN = (500_000.0, 1_500_000.0)
RES_M = 10.0
GRID = {"crs": UTM, "origin": UTM_ORIGIN, "resolution": RES_M, "dtype": "float32"}


def _write_pair(tmp_path, optical, sar, *, optical_names=("Red", "Green", "NIR"), **kw):
    o = write_geotiff(
        tmp_path / "opt.tif", array=optical, band_names=list(optical_names), **{**GRID, **kw}
    )
    s = write_geotiff(tmp_path / "sar.tif", array=sar, **{**GRID, **kw})
    return o, s


def _aligned(tmp_path, **kw):
    optical, sar = optical_sar_scene()
    o, s = _write_pair(tmp_path, optical, sar, **kw)
    return align_modalities(o, s), o, s


# --- modality alignment -----------------------------------------------------------


def test_align_modalities_keeps_each_band_set(tmp_path):
    """Fusion needs all optical bands AND the SAR band - no truncation to a common count."""
    aligned, _o, _s = _aligned(tmp_path)

    assert aligned.optical_band_count == 3
    assert aligned.sar_band_count == 1
    assert aligned.optical.shape == (3, aligned.height, aligned.width)
    assert aligned.sar.shape == (1, aligned.height, aligned.width)
    assert aligned.georeferenced is True
    assert aligned.crs == UTM
    assert aligned.pixel_area == pytest.approx(RES_M * RES_M)
    assert any("resampled both images" in op for op in aligned.operations)


def test_align_modalities_reprojects_a_crs_mismatch(tmp_path):
    optical, sar = optical_sar_scene(size=96)
    o = write_geotiff(tmp_path / "o.tif", array=optical, band_names=["Red", "Green", "NIR"], **GRID)
    s = write_geotiff(
        tmp_path / "s.tif", array=sar, crs="EPSG:4326", origin=(75.0, 13.57),
        resolution=0.0001, dtype="float32",
    )
    aligned = align_modalities(o, s)

    assert aligned.crs == UTM
    assert any("reprojected" in op for op in aligned.operations)
    assert aligned.optical_band_count == 3 and aligned.sar_band_count == 1


def test_align_modalities_rejects_non_overlapping(tmp_path):
    optical, sar = optical_sar_scene(size=64)
    o = write_geotiff(tmp_path / "o.tif", array=optical, band_names=["Red", "Green", "NIR"], **GRID)
    s = write_geotiff(
        tmp_path / "s.tif", array=sar, **{**GRID, "origin": (900_000.0, 1_900_000.0)}
    )
    with pytest.raises(AlignmentError) as exc:
        align_modalities(o, s)
    assert exc.value.code == "no_overlap"


def test_align_modalities_uses_coarser_resolution(tmp_path):
    optical, sar = optical_sar_scene(size=96)
    o = write_geotiff(tmp_path / "o.tif", array=optical, band_names=["Red", "Green", "NIR"], **GRID)
    # SAR at 20 m (coarser) than optical 10 m.
    s = write_geotiff(tmp_path / "s.tif", array=sar, **{**GRID, "resolution": RES_M * 2})
    aligned = align_modalities(o, s)
    assert aligned.pixel_area == pytest.approx((RES_M * 2) ** 2)
    assert any("coarser pixel size" in op for op in aligned.operations)


# --- cross-modal measurement ------------------------------------------------------


def test_cross_modal_reports_both_modalities_and_a_version(tmp_path):
    aligned, o, _s = _aligned(tmp_path)
    result = analyze_cross_modal(aligned, optical_roles=resolve_band_roles(read_metadata(o)))

    assert result.inconclusive is False
    assert result.algorithm_version == FUSION_ALGORITHM_VERSION
    assert result.optical.modality == "optical"
    assert result.sar.modality == "sar"
    assert 0.0 <= result.optical.mean_level <= 1.0
    assert 0.0 <= result.sar.mean_level <= 1.0


def test_sar_is_reported_uncalibrated_without_metadata(tmp_path):
    """No calibration tags -> must not claim sigma0/gamma0."""
    aligned, _o, _s = _aligned(tmp_path)
    result = analyze_cross_modal(aligned)
    assert result.sar_calibrated is False
    assert "uncalibrated" in result.sar_representation.lower()


def test_sar_calibration_is_recognised_from_metadata(tmp_path):
    optical, sar = optical_sar_scene(size=96)
    o = write_geotiff(tmp_path / "o.tif", array=optical, band_names=["Red", "Green", "NIR"], **GRID)
    s = write_geotiff(
        tmp_path / "s.tif", array=sar, tags={"processing": "sigma0 calibrated"}, **GRID
    )
    aligned = align_modalities(o, s)
    result = analyze_cross_modal(aligned, sar_tags=read_metadata(s).tags)
    assert result.sar_calibrated is True


def test_complementarity_is_measured_between_zero_and_one(tmp_path):
    aligned, o, _s = _aligned(tmp_path)
    result = analyze_cross_modal(aligned, optical_roles=resolve_band_roles(read_metadata(o)))

    assert result.correlation is not None
    assert -1.0 <= result.correlation <= 1.0
    assert result.complementarity == pytest.approx(1.0 - abs(result.correlation), abs=1e-3)


def test_correlation_withheld_when_a_modality_is_uniform(tmp_path):
    """A flat modality makes correlation meaningless; it must be withheld, not reported as 0."""
    optical, _sar = optical_sar_scene(size=96)
    flat_sar = np.full((1, 96, 96), 0.5, dtype="float32")
    o, s = _write_pair(tmp_path, optical, flat_sar)
    result = analyze_cross_modal(align_modalities(o, s))

    assert result.correlation is None
    assert result.complementarity is None
    assert any("nearly uniform" in w for w in result.warnings)


def test_water_agreement_is_measured_when_bands_declared(tmp_path):
    """Optical NDWI water and SAR dark surfaces should agree on the synthetic water strip."""
    aligned, o, _s = _aligned(tmp_path)
    result = analyze_cross_modal(aligned, optical_roles=resolve_band_roles(read_metadata(o)))

    assert result.water.available is True
    assert result.water.optical_water_fraction > 0
    assert result.water.sar_dark_fraction > 0
    assert result.water.agreement_iou > 0.2, "the water strip should agree across modalities"


def test_water_agreement_declined_without_declared_bands(tmp_path):
    optical, sar = optical_sar_scene(size=96)
    o, s = _write_pair(tmp_path, optical, sar, optical_names=())  # no band names
    result = analyze_cross_modal(
        align_modalities(o, s), optical_roles=resolve_band_roles(read_metadata(o))
    )
    assert result.water.available is False
    assert "NDWI" in result.water.reason


def test_sar_reveals_structure_region_is_found(tmp_path):
    """The structured block is rough in SAR but flat in optical -> a SAR-only region."""
    aligned, o, _s = _aligned(tmp_path)
    result = analyze_cross_modal(aligned, optical_roles=resolve_band_roles(read_metadata(o)))

    sar_regions = [r for r in result.regions if r.kind == "sar_structure"]
    assert sar_regions, "SAR should reveal at least one structured region the optical does not"
    region = sar_regions[0]
    assert region.pixel_count >= 30
    assert region.centroid_lonlat is not None  # georeferenced
    assert region.area_m2 is not None


def test_information_partition_sums_are_fractions(tmp_path):
    aligned, o, _s = _aligned(tmp_path)
    result = analyze_cross_modal(aligned, optical_roles=resolve_band_roles(read_metadata(o)))
    total = (
        result.optical_only_information
        + result.sar_only_information
        + result.shared_information
    )
    assert 0.0 <= total <= 1.0
    assert result.sar_only_information >= 0.0


def test_inconclusive_when_no_valid_overlap(tmp_path):
    blank_opt = np.zeros((3, 64, 64), dtype="float32")
    blank_sar = np.zeros((1, 64, 64), dtype="float32")
    o, s = _write_pair(tmp_path, blank_opt, blank_sar, nodata=0)
    result = analyze_cross_modal(align_modalities(o, s))
    assert result.inconclusive is True


def test_ndvi_only_from_declared_bands(tmp_path):
    aligned, o, _s = _aligned(tmp_path)
    with_bands = analyze_cross_modal(aligned, optical_roles=resolve_band_roles(read_metadata(o)))
    assert "ndvi_mean" in with_bands.optical.extra

    optical, sar = optical_sar_scene(size=96)
    o2, s2 = _write_pair(tmp_path, optical, sar, optical_names=())
    without = analyze_cross_modal(
        align_modalities(o2, s2), optical_roles=resolve_band_roles(read_metadata(o2))
    )
    assert "ndvi_mean" not in without.optical.extra


# --- question classification ------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Analyze these optical and SAR images together.", CrossModalQuestion.DESCRIBE),
        ("What complementary information can be obtained?", CrossModalQuestion.COMPLEMENTARY),
        ("What does SAR reveal that optical does not?", CrossModalQuestion.SAR_REVEALS),
        (
            "Which modality provides stronger evidence of vegetation?",
            CrossModalQuestion.WHICH_MODALITY,
        ),
        ("Is the water consistent between optical and SAR?", CrossModalQuestion.AGREEMENT),
        ("What is the average rainfall here?", CrossModalQuestion.UNSUPPORTED),
    ],
)
def test_cross_modal_question_classification(query, expected):
    kind, _topic = classify_cross_modal_question(query)
    assert kind is expected


def test_which_modality_extracts_topic():
    _kind, topic = classify_cross_modal_question("Which modality shows vegetation more strongly?")
    assert topic == "vegetation"


# --- rendering --------------------------------------------------------------------


def test_fusion_composite_renders_three_channels(tmp_path):
    aligned, _o, _s = _aligned(tmp_path)
    dest = tmp_path / "composite.png"
    w, h = render_fusion_composite(aligned, dest)
    assert dest.is_file()
    assert (w, h) == (aligned.width, aligned.height)
    arr = np.asarray(Image.open(dest).convert("RGB"))
    # Red channel (SAR) and green channel (optical) must differ somewhere, else it is not fused.
    assert int(np.abs(arr[:, :, 0].astype(int) - arr[:, :, 1].astype(int)).max()) > 20


def test_aligned_sar_renders(tmp_path):
    aligned, _o, _s = _aligned(tmp_path)
    dest = tmp_path / "sar.png"
    render_aligned_sar(aligned, dest)
    assert dest.is_file()


def test_complementary_map_renders(tmp_path):
    aligned, o, _s = _aligned(tmp_path)
    result = analyze_cross_modal(aligned, optical_roles=resolve_band_roles(read_metadata(o)))
    dest = tmp_path / "comp.png"
    render_complementary_map(aligned, result, dest)
    assert dest.is_file()


def test_optical_pil_is_rgb(tmp_path):
    aligned, _o, _s = _aligned(tmp_path)
    image = optical_pil(aligned)
    assert image.mode == "RGB"
    assert image.size == (aligned.width, aligned.height)


def test_non_georeferenced_equal_grids_compare_without_coordinates(tmp_path):
    optical, sar = optical_sar_scene(size=64)
    o = write_geotiff(tmp_path / "o.tif", array=optical, crs=None, dtype="float32")
    s = write_geotiff(tmp_path / "s.tif", array=sar, crs=None, dtype="float32")
    aligned = align_modalities(o, s)
    result = analyze_cross_modal(aligned, optical_roles=resolve_band_roles(read_metadata(o)))

    assert aligned.georeferenced is False
    assert result.area_units == "pixels"
    assert all(r.centroid_lonlat is None for r in result.regions)


def test_non_georeferenced_mismatched_dims_rejected(tmp_path):
    o = write_geotiff(tmp_path / "o.tif", width=64, height=64, bands=3, crs=None)
    s = write_geotiff(tmp_path / "s.tif", width=48, height=48, bands=1, crs=None)
    with pytest.raises(AlignmentError) as exc:
        align_modalities(o, s)
    assert exc.value.code == "dimension_mismatch"


def test_align_pair_unaffected_by_the_refactor(tmp_path):
    """Regression guard: the Phase 6 align_pair must still behave identically."""
    from geospatial import align_pair

    scene = textured_scene(size=64)
    a = write_geotiff(tmp_path / "a.tif", array=scene, **GRID)
    b = write_geotiff(tmp_path / "b.tif", array=scene, **GRID)
    aligned = align_pair(a, b)
    assert aligned.band_count == 3
    assert aligned.width == 64 and aligned.height == 64
    assert aligned.georeferenced is True
