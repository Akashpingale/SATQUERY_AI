"""Deterministic synthetic fixtures for Phase 9 engineering-validation benchmarks.

These are **synthetic** rasters generated from fixed seeds — no external download, no hidden
evaluation data. They exist to validate that the *production* specialists behave correctly and
respond to controlled inputs, and to provide labelled data for confidence calibration.

Results measured on these fixtures are **engineering validation**, never real-world benchmark
accuracy. Every harness and report says so explicitly. Real-world benchmarks (RSVQA, VRSBench,
CDVQA, SECOND) require their datasets and are reported separately.

Fixtures are written as valid GeoTIFFs (metric CRS, declared optical band roles, positive SAR
band) so they flow through the exact `geospatial` + `ml.inference` code paths the app uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

CRS = "EPSG:32643"  # UTM 43N (metres) — metric, so ground areas are meaningful
ORIGIN = (500_000.0, 1_500_000.0)
RES = 10.0
SIZE = 96
OPTICAL_BANDS = ("Red", "Green", "NIR")


def _profile(bands: int, dtype: str) -> dict:
    return {
        "driver": "GTiff",
        "width": SIZE,
        "height": SIZE,
        "count": bands,
        "dtype": dtype,
        "crs": CRS,
        "transform": from_origin(ORIGIN[0], ORIGIN[1], RES, RES),
    }


def _write(path: Path, array: np.ndarray, dtype: str, band_names: tuple[str, ...] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **_profile(array.shape[0], dtype)) as dst:
        dst.write(array.astype(dtype))
        for index, name in enumerate(band_names or (), start=1):
            dst.set_band_description(index, name)


def optical_scene(
    seed: int,
    *,
    left_right_gradient: float = 0.0,
    water_strip: bool = False,
    bright_block: tuple[slice, slice] | None = None,
    veg_scale: float = 1.0,
) -> np.ndarray:
    """A 3-band optical scene (Red, Green, NIR).

    Args:
        left_right_gradient: 0..1 strength of a left→right brightness ramp (for correlation
            control against SAR).
        water_strip: add an optically dark, NDWI-positive strip (NIR collapses, green stays).
        bright_block: a rectangular region raised in all bands (a "change"/structure block).
        veg_scale: multiplies NIR (vegetation) so NDVI can be moved between dates.
    """
    rng = np.random.default_rng(seed)
    cols = np.broadcast_to(np.arange(SIZE), (SIZE, SIZE)).astype("float64")
    ramp = (cols / (SIZE - 1)) * left_right_gradient

    red = np.clip(700 + 900 * ramp + rng.normal(0, 20, (SIZE, SIZE)), 1, 9000)
    green = np.clip(950 + 700 * ramp + rng.normal(0, 20, (SIZE, SIZE)), 1, 9000)
    nir = np.clip(veg_scale * (3000 + 400 * ramp) + rng.normal(0, 30, (SIZE, SIZE)), 1, 9000)

    if water_strip:
        water = slice(int(SIZE * 0.70), int(SIZE * 0.85))
        red[water, :] = 480
        green[water, :] = 1150
        nir[water, :] = 280  # NDWI = (G-NIR)/(G+NIR) positive

    if bright_block is not None:
        red[bright_block] = 2700
        green[bright_block] = 2600
        nir[bright_block] = 2500

    return np.stack([red, green, nir])


def sar_scene(
    seed: int,
    *,
    left_right_gradient: float = 0.0,
    dark_strip: bool = False,
    bright_block: tuple[slice, slice] | None = None,
) -> np.ndarray:
    """A single-band positive SAR-like backscatter scene on the same grid."""
    rng = np.random.default_rng(seed)
    cols = np.broadcast_to(np.arange(SIZE), (SIZE, SIZE)).astype("float64")
    ramp = (cols / (SIZE - 1)) * left_right_gradient
    sar = np.clip(0.20 + 1.4 * ramp + rng.gamma(2.0, 0.04, (SIZE, SIZE)), 1e-3, 6.0)

    if dark_strip:
        water = slice(int(SIZE * 0.70), int(SIZE * 0.85))
        sar[water, :] = 0.03  # smooth open water: dark in SAR

    if bright_block is not None:
        sar[bright_block] = np.clip(1.6 + rng.gamma(2.0, 0.15, sar[bright_block].shape), 0, 6)

    return sar[None, :, :]


# --- cross-modal scenarios --------------------------------------------------------


@dataclass
class CrossModalScenario:
    """One optical+SAR fixture with the qualitative property it should exhibit."""

    key: str
    optical_path: str
    sar_path: str
    #: Human-readable expected behaviour, checked by the harness.
    expectation: str


def build_cross_modal_fixtures(out_dir: str | Path) -> list[CrossModalScenario]:
    """Write four controlled optical+SAR fixtures exercising distinct fusion behaviours."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    block = (slice(int(SIZE * 0.15), int(SIZE * 0.42)), slice(int(SIZE * 0.20), int(SIZE * 0.52)))
    scenarios: list[CrossModalScenario] = []

    # 1. Correlated: optical brightness and SAR backscatter both ramp left->right.
    _write(
        out / "xm_corr_opt.tif", optical_scene(1, left_right_gradient=1.0), "uint16", OPTICAL_BANDS
    )
    _write(out / "xm_corr_sar.tif", sar_scene(2, left_right_gradient=1.0), "float32", None)
    scenarios.append(
        CrossModalScenario(
            "correlated",
            str(out / "xm_corr_opt.tif"),
            str(out / "xm_corr_sar.tif"),
            "positive correlation -> lower complementarity than the anticorrelated case",
        )
    )

    # 2. Anticorrelated: optical ramps up left->right, SAR ramps the opposite way.
    _write(
        out / "xm_anti_opt.tif", optical_scene(3, left_right_gradient=1.0), "uint16", OPTICAL_BANDS
    )
    sar_anti = sar_scene(4, left_right_gradient=1.0)[:, :, ::-1].copy()
    _write(out / "xm_anti_sar.tif", sar_anti, "float32", None)
    scenarios.append(
        CrossModalScenario(
            "anticorrelated",
            str(out / "xm_anti_opt.tif"),
            str(out / "xm_anti_sar.tif"),
            "negative correlation -> higher complementarity",
        )
    )

    # 3. SAR-only structure: optical near-flat, SAR has a bright rough block.
    _write(out / "xm_sar_opt.tif", optical_scene(5), "uint16", OPTICAL_BANDS)
    _write(out / "xm_sar_sar.tif", sar_scene(6, bright_block=block), "float32", None)
    scenarios.append(
        CrossModalScenario(
            "sar_structure",
            str(out / "xm_sar_opt.tif"),
            str(out / "xm_sar_sar.tif"),
            "SAR structured_fraction and SAR-only information exceed the flat-SAR case",
        )
    )

    # 4. Water agreement: optical NDWI-positive strip AND SAR dark strip at the same place.
    _write(out / "xm_water_opt.tif", optical_scene(7, water_strip=True), "uint16", OPTICAL_BANDS)
    _write(out / "xm_water_sar.tif", sar_scene(8, dark_strip=True), "float32", None)
    scenarios.append(
        CrossModalScenario(
            "water_agreement",
            str(out / "xm_water_opt.tif"),
            str(out / "xm_water_sar.tif"),
            "water agreement available with non-zero optical/SAR overlap",
        )
    )
    return scenarios


# --- change-detection scenarios ---------------------------------------------------


@dataclass
class ChangeScenario:
    """A bi-temporal pair with a known ground-truth changed mask."""

    key: str
    t1_path: str
    t2_path: str
    changed_mask: np.ndarray  # bool (H, W)
    changed: bool
    #: Fraction of the scene that changed by construction.
    changed_fraction: float = field(default=0.0)


def build_change_detection_fixtures(out_dir: str | Path) -> list[ChangeScenario]:
    """Write bi-temporal pairs: no-change, and blocks of increasing size (known masks)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenarios: list[ChangeScenario] = []

    base = optical_scene(100)

    def write_pair(key: str, t2: np.ndarray, mask: np.ndarray, changed: bool) -> None:
        _write(out / f"cd_{key}_t1.tif", base, "uint16", OPTICAL_BANDS)
        _write(out / f"cd_{key}_t2.tif", t2, "uint16", OPTICAL_BANDS)
        scenarios.append(
            ChangeScenario(
                key=key,
                t1_path=str(out / f"cd_{key}_t1.tif"),
                t2_path=str(out / f"cd_{key}_t2.tif"),
                changed_mask=mask,
                changed=changed,
                changed_fraction=round(float(mask.mean()), 5),
            )
        )

    # No change: T2 identical to T1 -> empty mask.
    write_pair("nochange", base.copy(), np.zeros((SIZE, SIZE), dtype=bool), False)

    # Blocks of increasing size at known locations.
    for key, (r0, r1, c0, c1) in {
        "small": (10, 22, 10, 22),
        "medium": (15, 45, 15, 45),
        "large": (10, 70, 10, 60),
    }.items():
        t2 = base.copy()
        block = (slice(r0, r1), slice(c0, c1))
        t2[0][block] = 2800
        t2[1][block] = 2700
        t2[2][block] = 2600
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        mask[block] = True
        write_pair(key, t2, mask, True)

    return scenarios


# --- change-VQA (occurrence) fixtures for accuracy + calibration ------------------


@dataclass
class OccurrenceSample:
    """A bi-temporal occurrence question with independent yes/no ground truth."""

    sample_id: str
    t1_path: str
    t2_path: str
    changed: bool  # ground truth: did a real radiometric change occur?
    magnitude: str  # "none" | "small" | "medium" | "large", for error analysis grouping


def build_change_vqa_fixtures(out_dir: str | Path) -> list[OccurrenceSample]:
    """Write graded bi-temporal pairs for change-occurrence VQA + calibration.

    A spread of change magnitudes (including none) makes the detector's confidence vary, so
    the calibration curve has signal. Ground truth is occurrence (yes/no), which is
    independent of the pipeline's answer phrasing, so accuracy is not graded against the
    system's own vocabulary.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = optical_scene(200)
    samples: list[OccurrenceSample] = []

    # Six no-change pairs (identical + tiny reseed noise below the noise floor) -> truth "no".
    for i in range(6):
        t2 = optical_scene(200 + (0 if i == 0 else 0))  # same seed -> identical scene
        key = f"nochange_{i}"
        _write(out / f"cvqa_{key}_t1.tif", base, "uint16", OPTICAL_BANDS)
        _write(out / f"cvqa_{key}_t2.tif", t2, "uint16", OPTICAL_BANDS)
        samples.append(
            OccurrenceSample(
                key, str(out / f"cvqa_{key}_t1.tif"), str(out / f"cvqa_{key}_t2.tif"), False, "none"
            )
        )

    # Graded change blocks -> truth "yes", with increasing size (confidence should rise).
    grades: list[tuple[str, tuple[int, int, int, int]]] = [
        ("small", (12, 22, 12, 22)),
        ("small", (60, 70, 20, 30)),
        ("medium", (12, 40, 12, 40)),
        ("medium", (40, 72, 30, 62)),
        ("large", (8, 78, 8, 60)),
        ("large", (10, 80, 20, 82)),
    ]
    for idx, (magnitude, (r0, r1, c0, c1)) in enumerate(grades):
        t2 = base.copy()
        block = (slice(r0, r1), slice(c0, c1))
        t2[0][block] = 2800
        t2[1][block] = 2700
        t2[2][block] = 2600
        key = f"change_{magnitude}_{idx}"
        _write(out / f"cvqa_{key}_t1.tif", base, "uint16", OPTICAL_BANDS)
        _write(out / f"cvqa_{key}_t2.tif", t2, "uint16", OPTICAL_BANDS)
        samples.append(
            OccurrenceSample(
                key,
                str(out / f"cvqa_{key}_t1.tif"),
                str(out / f"cvqa_{key}_t2.tif"),
                True,
                magnitude,
            )
        )

    return samples
