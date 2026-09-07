"""Optical + SAR cross-modal analysis.

The SIH requirement is *genuine* fusion: extract complementary information the two modalities
carry, not concatenate two separate captions. This module does that at the **feature level**.

Optical and SAR measure different physics. Optical records reflected sunlight (colour,
vegetation and water spectral cues, illumination-dependent, cloud-sensitive). SAR records radar
backscatter (surface roughness and structure, moisture, illumination-independent, speckled).
Because they measure different things, comparing them where they *agree* and where they
*differ* is itself the information — and neither image alone contains it.

Pipeline (all on the shared grid from :func:`geospatial.align_modalities`)::

    optical bands  -> brightness, spectral variance, texture, NDVI/NDWI (declared bands only)
    SAR band(s)    -> log-scaled backscatter, texture/roughness, dark/bright fractions
        |
    joint 2-channel feature space (optical brightness, SAR backscatter)
        -> cross-modal correlation  ->  complementarity = 1 - |correlation|
        -> information partition: optical-only / SAR-only / shared structure
        -> water agreement: optical NDWI-water  vs  SAR dark (smooth) surfaces
        -> coherent regions where SAR reveals structure the optical does not

Every number is measured from pixels. Honesty rules enforced here:

- **No calibrated units.** SAR is reported as *log-scaled, uncalibrated* backscatter unless the
  source metadata declares calibration; sigma0/gamma0 is never asserted otherwise.
- **No guessed bands.** NDVI/NDWI are computed only when the optical raster declares its bands.
- **No semantic SAR classes.** SAR evidence is stated as measured backscatter/roughness, never
  as "buildings"/"water" on SAR alone. Land-cover naming, when offered, is an optical-only
  zero-shot CLIP reading and is labelled indicative.
- **Complementarity needs signal.** If a modality is near-uniform, correlation-based numbers are
  withheld with a reason rather than reported as a spurious 0/1.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import numpy as np

from geospatial import BandRoles, normalized_difference, pixel_to_crs, to_lonlat
from geospatial.resample import AlignedModalities

#: Bumped when the analysis changes in a way that alters results.
FUSION_ALGORITHM_VERSION = "xmodal-1.0.0"

#: Window (pixels) for local-texture (roughness) estimation.
TEXTURE_WINDOW = 5
#: Percentiles for robust normalisation of a feature plane.
STRETCH_PERCENTILES = (2.0, 98.0)
#: A feature plane whose robust spread is below this (on the 0..1 scale) is "near-uniform":
#: correlation against it is meaningless, so cross-modal correlation is withheld.
MIN_FEATURE_SPREAD = 0.02
#: Local texture above this (0..1 scale) counts as "structured".
STRUCTURE_TEXTURE = 0.12
#: NDWI above this suggests open water (McFeeters convention: water is positive).
NDWI_WATER = 0.0
#: SAR backscatter below this percentile is treated as a smooth/dark surface.
SAR_DARK_PERCENTILE = 25.0
#: Minimum connected-region size (pixels) reported as a complementary region.
MIN_REGION_PIXELS = 30
#: Maximum complementary regions reported.
MAX_REGIONS = 5

#: Metadata tokens that would justify calling SAR calibrated. Absent -> uncalibrated.
_CALIBRATION_TOKENS = ("sigma0", "sigma_naught", "gamma0", "gamma_naught", "calibrat", "decibel")


class CrossModalQuestion(StrEnum):
    """What a cross-modal question asks for."""

    #: "describe using both", "analyze together" (default for an optical+SAR pair)
    DESCRIBE = "describe"
    #: "what complementary information", "what does SAR add"
    COMPLEMENTARY = "complementary"
    #: "what does SAR reveal that optical does not", "features visible in SAR"
    SAR_REVEALS = "sar_reveals"
    #: "which modality shows vegetation/water more strongly"
    WHICH_MODALITY = "which_modality"
    #: "is the water consistent between optical and SAR", "do they agree"
    AGREEMENT = "agreement"
    #: understood as cross-modal but not answerable from measurements
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ModalityFeatures:
    """Measured, per-modality summary. Optical and SAR carry different keys by nature."""

    modality: str
    #: Mean of the robust-normalised primary plane (brightness / backscatter), 0..1.
    mean_level: float
    #: Robust spread of that plane, 0..1 - how much dynamic range the modality shows here.
    spread: float
    #: Mean local texture (roughness/structure), 0..1.
    texture: float
    #: Fraction of the scene that is "structured" (texture above threshold).
    structured_fraction: float
    #: Extra measured quantities specific to the modality.
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "modality": self.modality,
            "mean_level": round(self.mean_level, 4),
            "spread": round(self.spread, 4),
            "texture": round(self.texture, 4),
            "structured_fraction": round(self.structured_fraction, 4),
            **{key: round(value, 4) for key, value in self.extra.items()},
        }


@dataclass(frozen=True)
class ComplementaryRegion:
    """A coherent area where one modality carries information the other does not."""

    index: int
    kind: str  # "sar_structure" | "water_agreement" | "optical_spectral"
    pixel_count: int
    box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    fraction_of_scene: float
    position: str
    area_m2: float | None = None
    centroid_lonlat: tuple[float, float] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "kind": self.kind,
            "pixel_count": self.pixel_count,
            "box": list(self.box),
            "fraction_of_scene": round(self.fraction_of_scene, 5),
            "position": self.position,
            "area_m2": round(self.area_m2, 1) if self.area_m2 is not None else None,
            "centroid_lonlat": (
                [round(self.centroid_lonlat[0], 6), round(self.centroid_lonlat[1], 6)]
                if self.centroid_lonlat
                else None
            ),
        }


@dataclass(frozen=True)
class WaterAgreement:
    """Cross-modal agreement on open water: optical NDWI vs SAR dark surfaces."""

    available: bool
    reason: str = ""
    optical_water_fraction: float = 0.0
    sar_dark_fraction: float = 0.0
    agreement_iou: float = 0.0
    agree_fraction: float = 0.0
    optical_only_fraction: float = 0.0
    sar_only_fraction: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "reason": self.reason,
            "optical_water_fraction": round(self.optical_water_fraction, 4),
            "sar_dark_fraction": round(self.sar_dark_fraction, 4),
            "agreement_iou": round(self.agreement_iou, 4),
            "agree_fraction": round(self.agree_fraction, 4),
            "optical_only_fraction": round(self.optical_only_fraction, 4),
            "sar_only_fraction": round(self.sar_only_fraction, 4),
        }


@dataclass
class CrossModalResult:
    """Everything the cross-modal analysis measured."""

    algorithm_version: str
    optical: ModalityFeatures
    sar: ModalityFeatures
    sar_representation: str
    sar_calibrated: bool

    #: Pearson correlation of optical brightness vs SAR backscatter over valid pixels, or
    #: None when a modality is near-uniform.
    correlation: float | None
    #: 1 - |correlation|: how much unique information the modalities carry relative to each
    #: other. High = strongly complementary. None when correlation is unavailable.
    complementarity: float | None

    #: Information partition over structured pixels (fractions of the scene).
    optical_only_information: float
    sar_only_information: float
    shared_information: float

    water: WaterAgreement
    regions: list[ComplementaryRegion] = field(default_factory=list)

    valid_fraction: float = 0.0
    georeferenced: bool = False
    area_units: str = "pixels"
    #: Optional optical-only land-cover reading (indicative). None when unavailable.
    optical_landcover: dict[str, object] | None = None
    warnings: list[str] = field(default_factory=list)
    inconclusive: bool = False
    inconclusive_reason: str = ""

    def stats_dict(self) -> dict[str, object]:
        return {
            "algorithm_version": self.algorithm_version,
            "sar_representation": self.sar_representation,
            "sar_calibrated": self.sar_calibrated,
            "correlation": None if self.correlation is None else round(self.correlation, 4),
            "complementarity": (
                None if self.complementarity is None else round(self.complementarity, 4)
            ),
            "optical_only_information": round(self.optical_only_information, 4),
            "sar_only_information": round(self.sar_only_information, 4),
            "shared_information": round(self.shared_information, 4),
            "valid_fraction": round(self.valid_fraction, 4),
            "georeferenced": self.georeferenced,
            "area_units": self.area_units,
            "optical": self.optical.as_dict(),
            "sar": self.sar.as_dict(),
            "water": self.water.as_dict(),
            "region_count": len(self.regions),
        }


# --- building blocks --------------------------------------------------------------


def _robust_normalize(plane: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Percentile-stretch a plane to 0..1 over valid pixels; invalid pixels set to 0."""
    values = plane[valid]
    if values.size == 0:
        return np.zeros(plane.shape, dtype="float32")
    lo, hi = np.percentile(values, STRETCH_PERCENTILES)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(plane.shape, dtype="float32")
    scaled = np.clip((plane - lo) / (hi - lo), 0.0, 1.0).astype("float32")
    scaled[~valid] = 0.0
    return scaled


def _local_texture(plane: np.ndarray, size: int = TEXTURE_WINDOW) -> np.ndarray:
    """Local standard deviation - a roughness/structure measure. 0..~1 for a 0..1 input."""
    from scipy import ndimage

    mean = ndimage.uniform_filter(plane, size=size, mode="reflect")
    mean_sq = ndimage.uniform_filter(plane * plane, size=size, mode="reflect")
    variance = np.clip(mean_sq - mean * mean, 0.0, None)
    return np.sqrt(variance).astype("float32")


def _spread(plane: np.ndarray, valid: np.ndarray) -> float:
    values = plane[valid]
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _sar_intensity(aligned: AlignedModalities, calibrated: bool) -> tuple[np.ndarray, str]:
    """Reduce SAR band(s) to one log-scaled backscatter plane.

    Multi-band SAR (e.g. VV+VH) is averaged in the linear domain, then log-scaled. SAR power
    is multiplicative, so the log domain is where its dynamic range and speckle behave
    additively - the standard representation. No calibration is claimed unless the metadata
    declared it.
    """
    linear = np.mean(aligned.sar, axis=0)
    positive = linear[np.isfinite(linear)]
    floor = 1e-6
    if positive.size and float(positive.min()) <= 0:
        # Shift so the log is defined without inventing a physical offset.
        floor = abs(float(positive.min())) + 1e-6
    log_plane = np.log10(np.clip(linear, 0, None) + floor)
    label = (
        "calibrated backscatter (log10, from source metadata)"
        if calibrated
        else "log10-scaled backscatter (uncalibrated; relative intensity only)"
    )
    return log_plane.astype("float32"), label


def _sar_is_calibrated(tags: dict[str, str] | None) -> bool:
    if not tags:
        return False
    blob = " ".join(f"{k} {v}" for k, v in tags.items()).lower()
    return any(token in blob for token in _CALIBRATION_TOKENS)


def _optical_brightness(aligned: AlignedModalities) -> np.ndarray:
    return np.mean(aligned.optical, axis=0).astype("float32")


def _optical_spectral_variance(aligned: AlignedModalities) -> np.ndarray:
    """Per-pixel variance across optical bands - high where colour discriminates land cover."""
    if aligned.optical_band_count < 2:
        return np.zeros((aligned.height, aligned.width), dtype="float32")
    return np.var(aligned.optical, axis=0).astype("float32")


def _correlation(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float | None:
    va, vb = a[valid], b[valid]
    if va.size < 8 or np.std(va) < 1e-6 or np.std(vb) < 1e-6:
        return None
    matrix = np.corrcoef(va, vb)
    value = float(matrix[0, 1])
    return value if np.isfinite(value) else None


def _crs_is_metric(crs: str | None) -> bool:
    if not crs:
        return False
    try:
        from pyproj import CRS

        info = CRS.from_user_input(crs)
        if info.is_geographic:
            return False
        unit = (info.axis_info[0].unit_name or "").lower() if info.axis_info else ""
        return unit in {"metre", "meter", "m"}
    except Exception:  # noqa: BLE001
        return False


def _position_label(col: float, row: float, width: int, height: int) -> str:
    vertical = "northern" if row < height / 3 else "southern" if row > 2 * height / 3 else "central"
    horizontal = "western" if col < width / 3 else "eastern" if col > 2 * width / 3 else "central"
    if vertical == "central" and horizontal == "central":
        return "centre"
    if vertical == "central":
        return f"{horizontal} side"
    if horizontal == "central":
        return f"{vertical} part"
    return f"{vertical} {horizontal} quadrant"


def _extract_regions(
    mask: np.ndarray, kind: str, aligned: AlignedModalities, metric: bool
) -> list[ComplementaryRegion]:
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return []
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    keep = [i for i, s in enumerate(sizes, start=1) if s >= MIN_REGION_PIXELS]
    keep.sort(key=lambda i: float(sizes[i - 1]), reverse=True)
    total = mask.shape[0] * mask.shape[1]

    regions: list[ComplementaryRegion] = []
    for order, label_id in enumerate(keep[:MAX_REGIONS], start=1):
        rows, cols = np.nonzero(labels == label_id)
        col_c, row_c = float(cols.mean()), float(rows.mean())
        lonlat = None
        if aligned.transform is not None and aligned.crs:
            x, y = pixel_to_crs(aligned.transform, col_c, row_c)
            lonlat = to_lonlat(aligned.crs, x, y)
        area = (
            float(len(rows) * aligned.pixel_area)
            if metric and aligned.pixel_area is not None
            else None
        )
        regions.append(
            ComplementaryRegion(
                index=order,
                kind=kind,
                pixel_count=int(len(rows)),
                box=(int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())),
                centroid=(col_c, row_c),
                fraction_of_scene=len(rows) / total if total else 0.0,
                position=_position_label(col_c, row_c, aligned.width, aligned.height),
                area_m2=area,
                centroid_lonlat=lonlat,
            )
        )
    return regions


def _water_agreement(
    ndwi: np.ndarray | None, sar_norm: np.ndarray, valid: np.ndarray
) -> WaterAgreement:
    """Compare optical water evidence (NDWI) against SAR dark (smooth) surfaces.

    This is a genuinely cross-modal measurement: smooth open water is bright-ish/blue in
    optical and *dark* in SAR (specular reflection away from the sensor). Where both agree,
    water is well supported by two independent physics; where they disagree, each modality is
    carrying something the other is not (e.g. SAR-dark roads/shadows, or optically-turbid
    water). None of this is claimed unless NDWI is actually computable.
    """
    if ndwi is None:
        return WaterAgreement(
            available=False,
            reason="NDWI is not computable (optical green/NIR bands not declared), so optical "
            "and SAR water evidence cannot be compared",
        )
    if not np.any(valid):
        return WaterAgreement(available=False, reason="no valid overlapping pixels")

    optical_water = (ndwi > NDWI_WATER) & valid
    dark_threshold = np.percentile(sar_norm[valid], SAR_DARK_PERCENTILE)
    sar_dark = (sar_norm <= dark_threshold) & valid

    union = optical_water | sar_dark
    intersection = optical_water & sar_dark
    valid_count = int(valid.sum())
    iou = float(intersection.sum() / union.sum()) if union.any() else 0.0
    return WaterAgreement(
        available=True,
        optical_water_fraction=float(optical_water.sum() / valid_count),
        sar_dark_fraction=float(sar_dark.sum() / valid_count),
        agreement_iou=iou,
        agree_fraction=float(intersection.sum() / valid_count),
        optical_only_fraction=float((optical_water & ~sar_dark).sum() / valid_count),
        sar_only_fraction=float((sar_dark & ~optical_water).sum() / valid_count),
    )


# --- the analysis -----------------------------------------------------------------


def analyze_cross_modal(
    aligned: AlignedModalities,
    *,
    optical_roles: BandRoles | None = None,
    sar_tags: dict[str, str] | None = None,
    min_valid_fraction: float = 0.20,
) -> CrossModalResult:
    """Measure complementary information between a co-registered optical/SAR pair.

    Args:
        aligned: Output of :func:`geospatial.align_modalities` (optical first, SAR second).
        optical_roles: Declared optical band roles, enabling NDVI/NDWI when present.
        sar_tags: SAR metadata tags, checked for calibration hints.
        min_valid_fraction: Below this shared-valid fraction the pair is inconclusive.
    """
    valid = aligned.valid
    valid_count = int(valid.sum())
    total = aligned.width * aligned.height
    valid_fraction = valid_count / total if total else 0.0
    metric = _crs_is_metric(aligned.crs)
    calibrated = _sar_is_calibrated(sar_tags)
    sar_plane, sar_label = _sar_intensity(aligned, calibrated)
    warnings: list[str] = []

    def inconclusive(reason: str) -> CrossModalResult:
        empty = ModalityFeatures("", 0.0, 0.0, 0.0, 0.0)
        return CrossModalResult(
            algorithm_version=FUSION_ALGORITHM_VERSION,
            optical=empty,
            sar=empty,
            sar_representation=sar_label,
            sar_calibrated=calibrated,
            correlation=None,
            complementarity=None,
            optical_only_information=0.0,
            sar_only_information=0.0,
            shared_information=0.0,
            water=WaterAgreement(available=False, reason=reason),
            valid_fraction=valid_fraction,
            georeferenced=aligned.georeferenced,
            area_units="square metres" if metric else "pixels",
            warnings=warnings,
            inconclusive=True,
            inconclusive_reason=reason,
        )

    if valid_count == 0:
        return inconclusive(
            "The optical and SAR images share no pixels with valid data in both, so nothing "
            "can be compared."
        )
    if valid_fraction < min_valid_fraction:
        return inconclusive(
            f"Only {valid_fraction:.0%} of the shared area carries valid data in both modalities,"
            f" below the {min_valid_fraction:.0%} needed for a reliable cross-modal comparison."
        )

    # --- per-modality feature planes, all on the shared grid --------------------
    optical_brightness = _robust_normalize(_optical_brightness(aligned), valid)
    sar_norm = _robust_normalize(sar_plane, valid)
    optical_texture = _local_texture(optical_brightness)
    sar_texture = _local_texture(sar_norm)
    spectral_variance = _robust_normalize(_optical_spectral_variance(aligned), valid)

    # NDVI / NDWI only from declared bands.
    ndvi = ndwi = None
    if optical_roles is not None:
        def band(index: int | None) -> np.ndarray | None:
            if index is None or index > aligned.optical.shape[0]:
                return None
            return aligned.optical[index - 1]

        if optical_roles.ndvi_available:
            red, nir = band(optical_roles.red), band(optical_roles.nir)
            if red is not None and nir is not None:
                ndvi = normalized_difference(nir, red)
        if optical_roles.ndwi_available:
            green, nir = band(optical_roles.green), band(optical_roles.nir)
            if green is not None and nir is not None:
                ndwi = normalized_difference(green, nir)

    optical_extra: dict[str, float] = {"spectral_variance": float(spectral_variance[valid].mean())}
    if ndvi is not None:
        optical_extra["ndvi_mean"] = float(ndvi[valid].mean())
    if ndwi is not None:
        optical_extra["ndwi_mean"] = float(ndwi[valid].mean())

    optical_features = ModalityFeatures(
        modality="optical",
        mean_level=float(optical_brightness[valid].mean()),
        spread=_spread(optical_brightness, valid),
        texture=float(optical_texture[valid].mean()),
        structured_fraction=float(
            ((optical_texture > STRUCTURE_TEXTURE) & valid).sum() / valid_count
        ),
        extra=optical_extra,
    )
    sar_features = ModalityFeatures(
        modality="sar",
        mean_level=float(sar_norm[valid].mean()),
        spread=_spread(sar_norm, valid),
        texture=float(sar_texture[valid].mean()),
        structured_fraction=float(((sar_texture > STRUCTURE_TEXTURE) & valid).sum() / valid_count),
        extra={
            "dark_fraction": float(
                (sar_norm <= np.percentile(sar_norm[valid], SAR_DARK_PERCENTILE)).sum()
                / valid_count
            ),
            "bright_fraction": float((sar_norm > 0.75).sum() / valid_count),
            "band_count": float(aligned.sar_band_count),
        },
    )

    # --- cross-modal correlation / complementarity ------------------------------
    if optical_features.spread < MIN_FEATURE_SPREAD or sar_features.spread < MIN_FEATURE_SPREAD:
        correlation = None
        complementarity = None
        warnings.append(
            "One modality is nearly uniform over the shared area, so a cross-modal correlation "
            "would be meaningless and is not reported."
        )
    else:
        correlation = _correlation(optical_brightness, sar_norm, valid)
        complementarity = None if correlation is None else round(1.0 - abs(correlation), 4)

    # --- information partition over structured pixels ---------------------------
    optical_struct = (optical_texture > STRUCTURE_TEXTURE) & valid
    sar_struct = (sar_texture > STRUCTURE_TEXTURE) & valid
    optical_only = optical_struct & ~sar_struct
    sar_only = sar_struct & ~optical_struct
    shared = optical_struct & sar_struct
    optical_only_information = float(optical_only.sum() / valid_count)
    sar_only_information = float(sar_only.sum() / valid_count)
    shared_information = float(shared.sum() / valid_count)

    water = _water_agreement(ndwi, sar_norm, valid)

    # --- complementary regions --------------------------------------------------
    regions: list[ComplementaryRegion] = []
    regions += _extract_regions(sar_only, "sar_structure", aligned, metric)
    if water.available and water.agree_fraction > 0:
        optical_water = (ndwi > NDWI_WATER) & valid if ndwi is not None else None
        if optical_water is not None:
            dark = sar_norm <= np.percentile(sar_norm[valid], SAR_DARK_PERCENTILE)
            regions += _extract_regions(
                optical_water & dark & valid, "water_agreement", aligned, metric
            )
    # Re-number across kinds so region indices are unique and stable.
    regions = [
        ComplementaryRegion(
            index=i + 1, kind=r.kind, pixel_count=r.pixel_count, box=r.box, centroid=r.centroid,
            fraction_of_scene=r.fraction_of_scene, position=r.position, area_m2=r.area_m2,
            centroid_lonlat=r.centroid_lonlat,
        )
        for i, r in enumerate(regions[:MAX_REGIONS])
    ]

    if not aligned.georeferenced:
        warnings.append(
            "Neither image carries usable georeferencing, so complementary regions are reported "
            "in pixels and cannot be tied to ground coordinates."
        )

    return CrossModalResult(
        algorithm_version=FUSION_ALGORITHM_VERSION,
        optical=optical_features,
        sar=sar_features,
        sar_representation=sar_label,
        sar_calibrated=calibrated,
        correlation=correlation,
        complementarity=complementarity,
        optical_only_information=optical_only_information,
        sar_only_information=sar_only_information,
        shared_information=shared_information,
        water=water,
        regions=regions,
        valid_fraction=valid_fraction,
        georeferenced=aligned.georeferenced,
        area_units="square metres" if metric else "pixels",
        warnings=warnings,
    )


# --- question interpretation -------------------------------------------------------


_COMPLEMENTARY_TERMS = (
    "complementary", "complement", "what does sar add", "additional information",
)
_SAR_REVEAL_TERMS = (
    "reveal", "less apparent", "not obvious", "not visible in optical", "hidden in optical",
    "visible in sar", "structure", "built structure", "roughness",
)
_AGREEMENT_TERMS = ("consistent", "agree", "agreement", "same in both", "match between")
_WHICH_TERMS = (
    "which modality", "which image", "stronger evidence", "which provides", "better for",
)
_DESCRIBE_TERMS = ("describe", "analyze", "analyse", "summar", "both images", "both modalities")

_VEGETATION_TERMS = ("vegetation", "vegetated", "forest", "green", "crop", "farmland")
_WATER_TERMS = ("water", "lake", "river", "flood", "wetland", "coast")
_BUILTUP_TERMS = ("built", "urban", "building", "structure", "settlement", "construction")


def classify_cross_modal_question(query: str) -> tuple[CrossModalQuestion, str | None]:
    """Route a cross-modal question. Returns ``(kind, topic)`` where topic is optional."""
    text = query.lower().strip()

    topic = None
    if any(t in text for t in _VEGETATION_TERMS):
        topic = "vegetation"
    elif any(t in text for t in _WATER_TERMS):
        topic = "water"
    elif any(t in text for t in _BUILTUP_TERMS):
        topic = "builtup"

    if any(t in text for t in _WHICH_TERMS) and topic:
        return CrossModalQuestion.WHICH_MODALITY, topic
    if any(t in text for t in _AGREEMENT_TERMS):
        return CrossModalQuestion.AGREEMENT, topic
    if any(t in text for t in _SAR_REVEAL_TERMS) and ("sar" in text or "radar" in text):
        return CrossModalQuestion.SAR_REVEALS, topic
    if any(t in text for t in _COMPLEMENTARY_TERMS):
        return CrossModalQuestion.COMPLEMENTARY, topic
    if any(t in text for t in _DESCRIBE_TERMS):
        return CrossModalQuestion.DESCRIBE, topic
    if re.search(r"\b(sar|radar|optical|both|fuse|fusion|combine)\b", text):
        return CrossModalQuestion.DESCRIBE, topic
    return CrossModalQuestion.UNSUPPORTED, topic


def cross_modal_stats(result: CrossModalResult) -> dict[str, object]:
    """Convenience for evidence payloads and the benchmark interface."""
    return {**result.stats_dict(), "regions": [r.as_dict() for r in result.regions]}


def _region_dict(result: CrossModalResult) -> list[dict[str, object]]:
    return [asdict(r) for r in result.regions]
