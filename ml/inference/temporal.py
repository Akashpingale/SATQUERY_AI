"""Bi-temporal change detection on co-registered imagery.

Method: **radiometrically normalised change vector analysis with data-driven
thresholding and spatial-coherence filtering.** Every stage is deterministic, measurable,
and inspectable - given the same two rasters and the same config, the same numbers come out.

Pipeline::

    aligned pair (geospatial.align_pair)
      -> misregistration estimate      (are we even comparing the same pixels?)
      -> relative radiometric normalisation  (remove illumination/atmosphere offsets)
      -> change magnitude              (CVA for optical, log-ratio for SAR)
      -> robust scaling                (comparable across scenes)
      -> threshold = max(Otsu, noise floor)
      -> binary opening                (drop speckle)
      -> connected components          (spatially coherent regions only)
      -> statistics, spectral index deltas, per-region geometry

Why not a pretrained deep change-detection network: see ``docs/phase6.md``. The short
version is that the available checkpoints (BIT, ChangeFormer, TinyCD) are distributed as
loose per-repo files rather than through a package index, and every one of them is trained
on 0.5 m aerial RGB *building* change (LEVIR-CD/WHU-CD). Applied to arbitrary satellite
resolutions they produce confident, plausible, wrong masks - the exact failure mode this
project forbids. This method instead measures something real at any resolution and reports
its own uncertainty.

Nothing here fabricates: a pixel is only marked changed when its measured magnitude clears
both thresholds and it belongs to a region large enough to survive coherence filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from geospatial import BandRoles, normalized_difference, pixel_to_crs, to_lonlat
from geospatial.resample import AlignedPair

#: Bumped whenever the algorithm changes in a way that alters results. Recorded in the
#: registry, the trace, and the evidence so a result can always be tied to a method.
ALGORITHM_VERSION = "cva-robust-1.1.0"

#: Below this, the robust noise floor is treated as degenerate: the unchanged population
#: matched too exactly for a spread-based significance test to mean anything. Expressed on the
#: scaled-magnitude axis, where 1.0 is the 99th percentile of measured magnitudes.
_DEGENERATE_FLOOR = 1e-4

#: One normalise-and-threshold pass:
#: ``(normalised_second, scaled_magnitude, normalisation_parameters, threshold, otsu,
#: noise_floor, threshold_method, effective_method)``.
#:
#: ``effective_method`` is carried out of the pass because :func:`change_magnitude` can fall
#: back from the log-ratio formulation when the data is not strictly positive. Dropping it
#: would let a result report a method it did not actually use.
_MeasurePass = tuple[
    np.ndarray, np.ndarray, list[dict[str, float]], float, float, float, str, "ChangeMethod"
]


class ChangeMethod(StrEnum):
    """Which magnitude formulation was used."""

    #: Euclidean distance across bands after radiometric normalisation. Optical default.
    CHANGE_VECTOR = "change_vector_analysis"
    #: |log(T2/T1)|, the standard detector for multiplicative SAR speckle noise.
    LOG_RATIO = "log_ratio"


@dataclass(frozen=True)
class ChangeConfig:
    """Every threshold the detector uses. No hidden constants.

    Defaults are chosen to be conservative: it is better to report "no significant change"
    than to report noise as change.
    """

    #: Multiplier on the robust noise spread (MAD-scaled) that sets the noise floor. 3.0 is
    #: roughly a 3-sigma criterion for Gaussian residuals.
    noise_sigma: float = 3.0
    #: Side of the square structuring element used for binary opening, in pixels. Removes
    #: isolated speckle while preserving coherent regions.
    opening_size: int = 3
    #: Connected regions smaller than this many pixels are discarded as noise.
    min_region_pixels: int = 25
    #: Maximum regions reported individually (largest first).
    max_reported_regions: int = 6
    #: Search radius in pixels for the misregistration estimate.
    max_shift_pixels: int = 3
    #: Longest side used for the misregistration correlation search, for speed.
    shift_search_max_side: int = 256
    #: Minimum share of residual decorrelation a shift must explain before the pair is
    #: called misregistered. Guards against a large local change producing a spurious
    #: correlation peak - see :attr:`Registration.shift_explains`.
    min_shift_explains: float = 0.5
    #: Largest measured shift still treated as tolerable jitter. Up to one common-grid pixel
    #: is normal co-registration error and the coherence filter absorbs the resulting edge
    #: noise. Beyond that, displacement artifacts dominate any measured change, so the
    #: result is reported inconclusive rather than presented as change.
    max_tolerable_shift_pixels: int = 1
    #: Changed-area fraction below which the scene is reported as no significant change.
    min_changed_fraction: float = 0.005
    #: Changed-area fraction above which a result is flagged as suspiciously global, which
    #: usually means radiometric or registration problems rather than real change.
    suspicious_changed_fraction: float = 0.60
    #: Minimum share of the common grid that must carry valid data in both images.
    min_valid_fraction: float = 0.20
    #: Peak normalised cross-correlation below which the pair looks poorly matched.
    low_correlation: float = 0.30

    def as_dict(self) -> dict[str, float | int]:
        return {
            "noise_sigma": self.noise_sigma,
            "opening_size": self.opening_size,
            "min_region_pixels": self.min_region_pixels,
            "max_shift_pixels": self.max_shift_pixels,
            "min_changed_fraction": self.min_changed_fraction,
            "suspicious_changed_fraction": self.suspicious_changed_fraction,
            "min_valid_fraction": self.min_valid_fraction,
            "low_correlation": self.low_correlation,
            "min_shift_explains": self.min_shift_explains,
            "max_tolerable_shift_pixels": self.max_tolerable_shift_pixels,
        }


@dataclass(frozen=True)
class Registration:
    """Measured registration quality between the two dates."""

    #: Best-correlating (row, col) shift in common-grid pixels. (0, 0) means aligned.
    peak_offset: tuple[int, int]
    #: Normalised cross-correlation at the best shift.
    peak_correlation: float
    #: Normalised cross-correlation with no shift applied.
    zero_correlation: float
    #: Search radius actually used, in common-grid pixels.
    searched_radius: int

    @property
    def aligned(self) -> bool:
        return self.peak_offset == (0, 0)

    @property
    def shift_pixels(self) -> int:
        """Chebyshev magnitude of the best-correlating shift."""
        return max(abs(self.peak_offset[0]), abs(self.peak_offset[1]))

    @property
    def shift_explains(self) -> float:
        """Share of the residual decorrelation that shifting removes.

        This is the discriminator between real misregistration and a spurious peak. A
        genuinely shifted pair starts highly correlated and shifting recovers almost all of
        the remaining difference, so this approaches 1.0. A pair that merely contains a large
        bright change also correlates slightly better under some shift, but that shift
        explains only a small share of a large decorrelation, so this stays near 0.

        Measured on synthetic pairs: a real 2 px shift scores ~1.00, while a strong local
        change with no shift scores ~0.09.
        """
        residual = 1.0 - self.zero_correlation
        if residual <= 1e-6:
            return 0.0
        return max(0.0, min(1.0, (self.peak_correlation - self.zero_correlation) / residual))


@dataclass(frozen=True)
class IndexChange:
    """Change in a spectral index, computed only when the bands were declared."""

    name: str
    available: bool
    reason: str = ""
    mean_before: float | None = None
    mean_after: float | None = None
    delta: float | None = None
    #: Mean delta restricted to pixels flagged as changed.
    delta_in_changed: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "reason": self.reason,
            "mean_before": self.mean_before,
            "mean_after": self.mean_after,
            "delta": self.delta,
            "delta_in_changed": self.delta_in_changed,
        }


@dataclass(frozen=True)
class ChangeRegion:
    """One spatially coherent changed area."""

    index: int
    pixel_count: int
    #: (x_min, y_min, x_max, y_max) on the common grid.
    box: tuple[int, int, int, int]
    centroid: tuple[float, float]  # (col, row)
    mean_magnitude: float
    max_magnitude: float
    #: Mean brightness change inside the region (positive = brighter at T2).
    brightness_delta: float
    fraction_of_scene: float
    #: Ground area in square metres, only when the CRS is projected in metres.
    area_m2: float | None = None
    #: (lon, lat) of the centroid, when georeferenced.
    centroid_lonlat: tuple[float, float] | None = None
    #: Compass description of where the region sits in the scene.
    position: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "pixel_count": self.pixel_count,
            "box": list(self.box),
            "centroid": [round(self.centroid[0], 1), round(self.centroid[1], 1)],
            "mean_magnitude": round(self.mean_magnitude, 4),
            "max_magnitude": round(self.max_magnitude, 4),
            "brightness_delta": round(self.brightness_delta, 4),
            "fraction_of_scene": round(self.fraction_of_scene, 5),
            "area_m2": round(self.area_m2, 1) if self.area_m2 is not None else None,
            "centroid_lonlat": (
                [round(self.centroid_lonlat[0], 6), round(self.centroid_lonlat[1], 6)]
                if self.centroid_lonlat
                else None
            ),
            "position": self.position,
        }


@dataclass
class ChangeResult:
    """Everything the detector measured. Consumed by the specialists and the renderer."""

    algorithm_version: str
    method: ChangeMethod
    config: ChangeConfig
    registration: Registration

    #: Robust-scaled magnitude, 0..1-ish, invalid pixels zeroed.
    magnitude: np.ndarray
    #: Final change mask after thresholding and coherence filtering.
    mask: np.ndarray
    valid: np.ndarray

    threshold: float
    threshold_method: str
    otsu_threshold: float
    noise_floor: float

    changed_pixels: int
    valid_pixels: int
    changed_fraction: float
    #: Changed pixels before coherence filtering, for a measurable noise-suppression figure.
    raw_changed_pixels: int
    regions: list[ChangeRegion] = field(default_factory=list)
    region_count: int = 0

    #: Mean brightness change over changed pixels (positive = brighter at T2).
    brightness_delta: float = 0.0
    #: Per-band mean change over changed pixels.
    band_deltas: list[float] = field(default_factory=list)
    #: Radiometric normalisation gains/offsets actually applied, per band.
    normalization: list[dict[str, float]] = field(default_factory=list)
    #: Whether the normalisation was refitted on presumed-unchanged pixels (second pass).
    normalization_refined: bool = False

    indices: list[IndexChange] = field(default_factory=list)
    georeferenced: bool = False
    area_units: str = "pixels"
    changed_area_m2: float | None = None
    #: Findings that must reach the user; each is a real measured condition.
    warnings: list[str] = field(default_factory=list)
    #: True when the evidence does not support any change claim.
    inconclusive: bool = False
    inconclusive_reason: str = ""

    @property
    def significant(self) -> bool:
        """Whether a change claim is supported at all."""
        return (
            not self.inconclusive
            and self.changed_fraction >= self.config.min_changed_fraction
            and self.region_count > 0
        )

    @property
    def coherence(self) -> float:
        """Share of above-threshold pixels that survived coherence filtering.

        Low values mean the raw signal was mostly scattered speckle rather than real
        coherent change.
        """
        if self.raw_changed_pixels <= 0:
            return 0.0
        return min(1.0, self.changed_pixels / self.raw_changed_pixels)

    def index_by_name(self, name: str) -> IndexChange | None:
        for item in self.indices:
            if item.name == name:
                return item
        return None

    def direction_for(self, index_name: str) -> str | None:
        """``"increase"``, ``"decrease"``, ``"no change"``, or None when unmeasurable.

        Lives here rather than in the wording layer so the serving path and a benchmark
        harness derive the direction from exactly the same measurement.
        """
        index = self.index_by_name(index_name)
        if index is None or not index.available or index.delta is None:
            return None
        if index.delta > 0:
            return "increase"
        if index.delta < 0:
            return "decrease"
        return "no change"

    def stats_dict(self) -> dict[str, object]:
        """Measured statistics, for evidence payloads."""
        return {
            "algorithm": self.method.value,
            "algorithm_version": self.algorithm_version,
            "threshold": round(self.threshold, 4),
            "threshold_method": self.threshold_method,
            "otsu_threshold": round(self.otsu_threshold, 4),
            "noise_floor": round(self.noise_floor, 4),
            "changed_pixels": self.changed_pixels,
            "valid_pixels": self.valid_pixels,
            "changed_fraction": round(self.changed_fraction, 5),
            "changed_percent": round(self.changed_fraction * 100, 2),
            "raw_changed_pixels": self.raw_changed_pixels,
            "coherence": round(self.coherence, 4),
            "region_count": self.region_count,
            "brightness_delta": round(self.brightness_delta, 4),
            "band_deltas": [round(value, 4) for value in self.band_deltas],
            "normalization_refined": self.normalization_refined,
            "registration_peak_offset": list(self.registration.peak_offset),
            "registration_peak_correlation": round(self.registration.peak_correlation, 4),
            "registration_zero_correlation": round(self.registration.zero_correlation, 4),
            "registration_shift_explains": round(self.registration.shift_explains, 4),
            "registration_shift_pixels": self.registration.shift_pixels,
            "area_units": self.area_units,
            "changed_area_m2": (
                round(self.changed_area_m2, 1) if self.changed_area_m2 is not None else None
            ),
            "georeferenced": self.georeferenced,
            "significant": self.significant,
        }


# --- building blocks --------------------------------------------------------------


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method: the histogram split that maximises between-class variance.

    **Reported, not used as the threshold.** Kept because it is a familiar reference value
    and publishing it next to the threshold that *was* used makes the choice auditable. It is
    not the decision criterion: Otsu always returns some split, including for an unchanged
    scene of pure noise, and when one change dominates a scene its split rises to separate
    that change from everything else, hiding weaker real change. See :func:`detect_change`.

    Returns 0.0 for empty input.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return float(lo)

    counts, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    counts = counts.astype("float64")
    total = counts.sum()
    if total <= 0:
        return float(lo)

    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(counts)
    weight_fg = total - weight_bg
    usable = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(usable):
        return float(lo)

    cumulative_mean = np.cumsum(counts * centers)
    total_mean = cumulative_mean[-1]
    mean_bg = np.divide(
        cumulative_mean, weight_bg, out=np.zeros_like(cumulative_mean), where=weight_bg > 0
    )
    mean_fg = np.divide(
        total_mean - cumulative_mean,
        weight_fg,
        out=np.zeros_like(cumulative_mean),
        where=weight_fg > 0,
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~usable] = -np.inf
    return float(centers[int(np.argmax(between))])


def robust_spread(values: np.ndarray) -> tuple[float, float]:
    """Return ``(median, MAD-scaled sigma)``.

    The median absolute deviation is scaled by 1.4826 so it estimates the standard
    deviation of a Gaussian. Robust statistics are essential here: real change is an
    outlier population, and a plain mean/std would be dragged by it.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median, mad * 1.4826


def radiometric_normalize(
    reference: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    sample: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Linearly rescale ``target`` so each band matches ``reference``'s level and spread.

    Two acquisitions of the same place differ in illumination, sun angle, and atmosphere.
    Without this step those differences dominate the measured "change". This is relative
    (image-to-image) normalisation, not absolute atmospheric correction - we have no
    atmospheric parameters, so no absolute claim is made.

    **Robust statistics, deliberately.** The level and spread come from the median and a
    MAD-scaled sigma, not the mean and standard deviation. Real change is an outlier
    population, and mean/std normalisation lets it drive the fit: a measured case where the
    near-infrared band collapsed over 17 % of a scene had its own signal rescaled away,
    reporting 0 % change. Median/MAD is unaffected by an outlier minority, so the change
    survives normalisation.

    Args:
        reference: Earlier image, ``(bands, H, W)``.
        target: Later image to rescale.
        valid: Pixels carrying data in both images.
        sample: Optional subset of ``valid`` to fit on. Passing the presumed no-change
            pixels here is what makes the second pass more accurate than the first.

    Returns ``(normalised_target, per_band_parameters)``.
    """
    normalised = target.astype("float32", copy=True)
    parameters: list[dict[str, float]] = []
    fit_on = valid if sample is None else sample

    for band in range(target.shape[0]):
        ref_band = reference[band][fit_on]
        tgt_band = target[band][fit_on]
        if ref_band.size == 0 or tgt_band.size == 0:
            parameters.append({"gain": 1.0, "offset": 0.0})
            continue

        ref_level, ref_spread = robust_spread(ref_band)
        tgt_level, tgt_spread = robust_spread(tgt_band)

        # A near-constant band carries no contrast to match; leave its scale alone rather
        # than dividing by ~0 and amplifying noise into fake change.
        gain = (ref_spread / tgt_spread) if tgt_spread > 1e-6 else 1.0
        offset = ref_level - gain * tgt_level
        normalised[band] = normalised[band] * gain + offset
        parameters.append({"gain": round(gain, 6), "offset": round(offset, 6)})

    return normalised, parameters


def _to_intensity(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Collapse a multi-band array to a single normalised intensity plane."""
    intensity = np.mean(data, axis=0)
    finite = intensity[valid] if np.any(valid) else intensity[np.isfinite(intensity)]
    if finite.size == 0:
        return np.zeros(intensity.shape, dtype="float32")
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros(intensity.shape, dtype="float32")
    return np.clip((intensity - lo) / (hi - lo), 0.0, 1.0).astype("float32")


def _decimate(plane: np.ndarray, max_side: int) -> np.ndarray:
    """Cheap strided decimation, for the shift search only."""
    longest = max(plane.shape)
    if longest <= max_side:
        return plane
    step = int(np.ceil(longest / max_side))
    return plane[::step, ::step]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-normalised cross-correlation of two equal-shaped planes."""
    if a.size == 0 or b.size == 0:
        return 0.0
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = float(np.sqrt((a_centered**2).sum() * (b_centered**2).sum()))
    if denominator <= 1e-12:
        return 0.0
    return float((a_centered * b_centered).sum() / denominator)


def estimate_misregistration(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
    config: ChangeConfig,
) -> Registration:
    """Measure whether the two dates actually line up.

    Searches small integer shifts and reports which one correlates best. A peak away from
    (0, 0) means the pair is misregistered by that many pixels, which makes every edge in
    the scene look like change. This is a *measurement*, so the resulting warning and
    confidence penalty are earned rather than assumed.
    """
    plane_a = _decimate(_to_intensity(first, valid), config.shift_search_max_side)
    plane_b = _decimate(_to_intensity(second, valid), config.shift_search_max_side)

    radius = max(0, int(config.max_shift_pixels))
    height, width = plane_a.shape
    if height <= 2 * radius + 2 or width <= 2 * radius + 2:
        radius = 0

    zero = _ncc(plane_a, plane_b)
    best_offset = (0, 0)
    best_score = zero

    for d_row in range(-radius, radius + 1):
        for d_col in range(-radius, radius + 1):
            if d_row == 0 and d_col == 0:
                continue
            # Crop both planes to the region they share under this shift.
            row_a = slice(max(0, d_row), height + min(0, d_row))
            col_a = slice(max(0, d_col), width + min(0, d_col))
            row_b = slice(max(0, -d_row), height + min(0, -d_row))
            col_b = slice(max(0, -d_col), width + min(0, -d_col))
            score = _ncc(plane_a[row_a, col_a], plane_b[row_b, col_b])
            if score > best_score:
                best_score = score
                best_offset = (d_row, d_col)

    return Registration(
        peak_offset=best_offset,
        peak_correlation=best_score,
        zero_correlation=zero,
        searched_radius=radius,
    )


def change_magnitude(
    first: np.ndarray, second: np.ndarray, method: ChangeMethod
) -> tuple[np.ndarray, ChangeMethod]:
    """Per-pixel change magnitude.

    Returns ``(magnitude, method_actually_used)`` - the method can fall back when the data
    does not support the requested formulation.
    """
    if method is ChangeMethod.LOG_RATIO:
        # Log-ratio is the textbook SAR detector because speckle is multiplicative: a ratio
        # turns it into an additive, roughly symmetric quantity. It needs strictly positive
        # values, so a band containing zeros or negatives falls back to differencing.
        if float(np.nanmin(first)) > 0.0 and float(np.nanmin(second)) > 0.0:
            ratio = np.log(second + 1e-6) - np.log(first + 1e-6)
            return np.abs(ratio).mean(axis=0).astype("float32"), ChangeMethod.LOG_RATIO
        method = ChangeMethod.CHANGE_VECTOR

    difference = second - first
    magnitude = np.sqrt(np.sum(difference.astype("float64") ** 2, axis=0))
    return magnitude.astype("float32"), ChangeMethod.CHANGE_VECTOR


def _scale_magnitude(magnitude: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rescale magnitude so thresholds mean the same thing across scenes.

    Divided by the 99th percentile of valid magnitudes: robust to a few extreme pixels while
    keeping the bulk of the distribution in 0..1.
    """
    values = magnitude[valid]
    if values.size == 0:
        return np.zeros(magnitude.shape, dtype="float32")
    reference = float(np.percentile(values, 99.0))
    if not np.isfinite(reference) or reference <= 0:
        reference = float(values.max()) if values.max() > 0 else 1.0
    scaled = magnitude / reference
    scaled[~valid] = 0.0
    return np.clip(scaled, 0.0, 4.0).astype("float32")


def _binary_open(mask: np.ndarray, size: int) -> np.ndarray:
    """Morphological opening: erosion then dilation, removing speckle."""
    if size < 2:
        return mask
    from scipy import ndimage

    structure = np.ones((size, size), dtype=bool)
    return ndimage.binary_opening(mask, structure=structure)


def _position_label(col: float, row: float, width: int, height: int) -> str:
    """Compass description of a point in the scene, for grounded wording."""
    vertical = "northern" if row < height / 3 else "southern" if row > 2 * height / 3 else "central"
    horizontal = "western" if col < width / 3 else "eastern" if col > 2 * width / 3 else "central"
    if vertical == "central" and horizontal == "central":
        return "centre"
    if vertical == "central":
        return f"{horizontal} side"
    if horizontal == "central":
        return f"{vertical} part"
    return f"{vertical} {horizontal.replace('ern', 'ern')} quadrant"


def _crs_is_metric(crs: str | None) -> bool:
    """Whether the CRS measures in metres, so an area in m^2 is meaningful."""
    if not crs:
        return False
    try:
        from pyproj import CRS

        info = CRS.from_user_input(crs)
        if info.is_geographic:
            return False
        unit = (info.axis_info[0].unit_name or "").lower() if info.axis_info else ""
        return unit in {"metre", "meter", "m"}
    except Exception:  # noqa: BLE001 - an unusable CRS supports no area claim
        return False


def _extract_regions(
    mask: np.ndarray,
    magnitude: np.ndarray,
    brightness_change: np.ndarray,
    aligned: AlignedPair,
    config: ChangeConfig,
) -> tuple[list[ChangeRegion], np.ndarray, int]:
    """Label coherent regions, drop small ones, and describe what survives.

    Returns ``(regions, filtered_mask, total_region_count)``.
    """
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return [], np.zeros(mask.shape, dtype=bool), 0

    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    keep_ids = [
        index for index, size in enumerate(sizes, start=1) if size >= config.min_region_pixels
    ]
    if not keep_ids:
        return [], np.zeros(mask.shape, dtype=bool), 0

    filtered = np.isin(labels, keep_ids)
    total_pixels = mask.shape[0] * mask.shape[1]
    metric = _crs_is_metric(aligned.crs)

    # Largest first: the biggest coherent region is what a user asks about.
    keep_ids.sort(key=lambda index: float(sizes[index - 1]), reverse=True)

    regions: list[ChangeRegion] = []
    for order, label_id in enumerate(keep_ids[: config.max_reported_regions], start=1):
        selection = labels == label_id
        rows, cols = np.nonzero(selection)
        pixel_count = int(selection.sum())
        centroid_col = float(cols.mean())
        centroid_row = float(rows.mean())

        centroid_lonlat = None
        if aligned.transform is not None and aligned.crs:
            x, y = pixel_to_crs(aligned.transform, centroid_col, centroid_row)
            centroid_lonlat = to_lonlat(aligned.crs, x, y)

        area_m2 = None
        if metric and aligned.pixel_area is not None:
            area_m2 = float(pixel_count * aligned.pixel_area)

        regions.append(
            ChangeRegion(
                index=order,
                pixel_count=pixel_count,
                box=(int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())),
                centroid=(centroid_col, centroid_row),
                mean_magnitude=float(magnitude[selection].mean()),
                max_magnitude=float(magnitude[selection].max()),
                brightness_delta=float(brightness_change[selection].mean()),
                fraction_of_scene=pixel_count / total_pixels if total_pixels else 0.0,
                area_m2=area_m2,
                centroid_lonlat=centroid_lonlat,
                position=_position_label(
                    centroid_col, centroid_row, aligned.width, aligned.height
                ),
            )
        )

    return regions, filtered, len(keep_ids)


def _index_change(
    name: str,
    available: bool,
    reason: str,
    before: np.ndarray | None,
    after: np.ndarray | None,
    valid: np.ndarray,
    mask: np.ndarray,
) -> IndexChange:
    if not available or before is None or after is None or not np.any(valid):
        return IndexChange(name=name, available=False, reason=reason)

    mean_before = float(before[valid].mean())
    mean_after = float(after[valid].mean())
    delta_in_changed = None
    changed = mask & valid
    if np.any(changed):
        delta_in_changed = float((after[changed] - before[changed]).mean())

    return IndexChange(
        name=name,
        available=True,
        mean_before=round(mean_before, 4),
        mean_after=round(mean_after, 4),
        delta=round(mean_after - mean_before, 4),
        delta_in_changed=round(delta_in_changed, 4) if delta_in_changed is not None else None,
    )


def _compute_indices(
    aligned: AlignedPair,
    roles_a: BandRoles | None,
    roles_b: BandRoles | None,
    valid: np.ndarray,
    mask: np.ndarray,
) -> list[IndexChange]:
    """NDVI and NDWI deltas, computed only where both rasters declare the bands.

    Indices are computed from the **raw** aligned reflectance, not the radiometrically
    normalised arrays: normalisation is a per-band linear rescale, and a band ratio is not
    invariant under it, so normalised inputs would give a distorted index.
    """
    undeclared = (
        "the raster does not declare which band is which, and guessing band order would "
        "produce a plausible but unfounded index"
    )
    if roles_a is None or roles_b is None:
        return [
            IndexChange(name="NDVI", available=False, reason=undeclared),
            IndexChange(name="NDWI", available=False, reason=undeclared),
        ]

    def band(data: np.ndarray, index: int | None) -> np.ndarray | None:
        if index is None or index > data.shape[0]:
            return None
        return data[index - 1]

    results: list[IndexChange] = []

    def all_present(*planes: np.ndarray | None) -> bool:
        # Explicit identity checks: `None not in (...)` would compare element-wise against
        # the arrays and raise.
        return all(plane is not None for plane in planes)

    ndvi_ready = roles_a.ndvi_available and roles_b.ndvi_available
    red_a, nir_a = band(aligned.first, roles_a.red), band(aligned.first, roles_a.nir)
    red_b, nir_b = band(aligned.second, roles_b.red), band(aligned.second, roles_b.nir)
    if ndvi_ready and all_present(red_a, nir_a, red_b, nir_b):
        results.append(
            _index_change(
                "NDVI",
                True,
                "",
                normalized_difference(nir_a, red_a),
                normalized_difference(nir_b, red_b),
                valid,
                mask,
            )
        )
    else:
        results.append(
            IndexChange(
                name="NDVI",
                available=False,
                reason="red and near-infrared bands are not both declared" if not ndvi_ready
                else "the declared band indexes exceed the compared band count",
            )
        )

    ndwi_ready = roles_a.ndwi_available and roles_b.ndwi_available
    green_a = band(aligned.first, roles_a.green)
    green_b = band(aligned.second, roles_b.green)
    if ndwi_ready and all_present(green_a, nir_a, green_b, nir_b):
        results.append(
            _index_change(
                "NDWI",
                True,
                "",
                normalized_difference(green_a, nir_a),
                normalized_difference(green_b, nir_b),
                valid,
                mask,
            )
        )
    else:
        results.append(
            IndexChange(
                name="NDWI",
                available=False,
                reason="green and near-infrared bands are not both declared" if not ndwi_ready
                else "the declared band indexes exceed the compared band count",
            )
        )

    return results


# --- the detector -----------------------------------------------------------------


def detect_change(
    aligned: AlignedPair,
    *,
    is_sar: bool = False,
    roles_first: BandRoles | None = None,
    roles_second: BandRoles | None = None,
    config: ChangeConfig | None = None,
) -> ChangeResult:
    """Measure change between two co-registered dates.

    Args:
        aligned: Output of :func:`geospatial.align_pair`.
        is_sar: Use the SAR log-ratio formulation instead of change vector analysis.
        roles_first: Declared band roles of the earlier image, if any.
        roles_second: Declared band roles of the later image, if any.
        config: Threshold configuration; defaults to :class:`ChangeConfig`.

    Returns:
        A :class:`ChangeResult`. When the evidence cannot support a change claim the result
        is marked ``inconclusive`` with a reason instead of reporting a mask anyway.
    """
    config = config or ChangeConfig()
    valid = aligned.valid
    valid_pixels = int(valid.sum())
    total_pixels = aligned.width * aligned.height
    warnings: list[str] = []

    method = ChangeMethod.LOG_RATIO if is_sar else ChangeMethod.CHANGE_VECTOR
    registration = estimate_misregistration(aligned.first, aligned.second, valid, config)

    def inconclusive(reason: str) -> ChangeResult:
        return ChangeResult(
            algorithm_version=ALGORITHM_VERSION,
            method=method,
            config=config,
            registration=registration,
            magnitude=np.zeros((aligned.height, aligned.width), dtype="float32"),
            mask=np.zeros((aligned.height, aligned.width), dtype=bool),
            valid=valid,
            threshold=0.0,
            threshold_method="none",
            otsu_threshold=0.0,
            noise_floor=0.0,
            changed_pixels=0,
            valid_pixels=valid_pixels,
            changed_fraction=0.0,
            raw_changed_pixels=0,
            georeferenced=aligned.georeferenced,
            area_units="square metres" if _crs_is_metric(aligned.crs) else "pixels",
            warnings=warnings,
            inconclusive=True,
            inconclusive_reason=reason,
        )

    if valid_pixels == 0:
        return inconclusive(
            "The two images share no pixels that carry valid data in both, so nothing can "
            "be compared."
        )

    valid_fraction = valid_pixels / total_pixels if total_pixels else 0.0
    if valid_fraction < config.min_valid_fraction:
        return inconclusive(
            f"Only {valid_fraction:.0%} of the shared area carries valid data in both images, "
            f"below the {config.min_valid_fraction:.0%} needed for a reliable comparison."
        )

    misregistered = (
        not registration.aligned and registration.shift_explains >= config.min_shift_explains
    )
    if misregistered:
        warnings.append(
            f"The images correlate best when shifted by {registration.peak_offset[1]} column(s) "
            f"and {registration.peak_offset[0]} row(s) rather than at zero offset, and that "
            f"shift explains {registration.shift_explains:.0%} of their remaining difference. "
            "The pair carries residual misregistration, so feature edges can appear as change."
        )
        if registration.shift_pixels > config.max_tolerable_shift_pixels:
            return inconclusive(
                f"The two images are misregistered by about {registration.shift_pixels} pixels "
                f"on the shared grid ({registration.peak_offset[1]} across, "
                f"{registration.peak_offset[0]} down). At that displacement, differences caused "
                "by shifting cannot be separated from real change, so no change result is "
                "reported. Re-run with co-registered imagery."
            )

    if registration.peak_correlation < config.low_correlation:
        warnings.append(
            f"The two images correlate only {registration.peak_correlation:.2f} at best. They "
            "may not show the same scene, or the scene changed so much that structural "
            "correspondence is lost; either way, per-pixel change is less reliable."
        )

    # --- normalise, then measure -------------------------------------------------
    def measure(sample: np.ndarray | None) -> _MeasurePass:
        """One full normalise-and-threshold pass. Returns the intermediate quantities."""
        normalised, parameters = radiometric_normalize(
            aligned.first, aligned.second, valid, sample=sample
        )
        raw, effective = change_magnitude(aligned.first, normalised, method)
        scaled = _scale_magnitude(raw, valid)

        pool = scaled[valid]
        otsu_value = otsu_threshold(pool)
        median, sigma = robust_spread(pool)
        floor = median + config.noise_sigma * sigma

        # The robust noise floor is the criterion: "is this pixel an outlier against the
        # unchanged population?" is exactly the question being asked, and MAD-based statistics
        # are unmoved by however much real change is present.
        #
        # Otsu's split is computed and reported for transparency but deliberately does NOT set
        # the threshold. Two formulations were tried and rejected on measurement:
        #   * max(otsu, floor) - when one change dominates a scene radiometrically, Otsu climbs
        #     to separate *that* change from everything else. A measured case had a large bright
        #     block push the split to 0.25 and hide a real dark patch completely.
        #   * otsu as the degenerate-case fallback - same failure, for the same reason.
        #
        # When the unchanged population has effectively zero spread (bit-identical pixels, as
        # happens with derived or synthetic imagery) there is no distribution to test against,
        # and any non-zero difference is by definition an outlier. The threshold then drops to
        # the smallest meaningful magnitude, and coherence filtering plus the minimum region
        # size provide the protection instead.
        if floor > _DEGENERATE_FLOOR:
            return (
                normalised,
                scaled,
                parameters,
                float(floor),
                otsu_value,
                floor,
                "noise_floor",
                effective,
            )
        return (
            normalised,
            scaled,
            parameters,
            _DEGENERATE_FLOOR,
            otsu_value,
            floor,
            "any measurable difference (unchanged pixels are identical)",
            effective,
        )

    (
        normalised_second,
        magnitude,
        normalization,
        threshold,
        otsu,
        noise_floor,
        threshold_method,
        method,
    ) = measure(None)
    first_pass_mask = (magnitude > threshold) & valid

    # Second pass: refit the normalisation on the pixels the first pass did NOT flag. Fitting
    # on presumed-unchanged ground is the standard way to keep real change from biasing the
    # radiometric fit. Skipped when too little of the scene looks unchanged, since then the
    # sample would be unrepresentative.
    unchanged = valid & ~first_pass_mask
    refined = False
    if int(unchanged.sum()) >= max(64, int(0.30 * valid_pixels)):
        (
            normalised_second,
            magnitude,
            normalization,
            threshold,
            otsu,
            noise_floor,
            threshold_method,
            method,
        ) = measure(unchanged)
        refined = True

    raw_mask = (magnitude > threshold) & valid
    raw_changed = int(raw_mask.sum())

    opened = _binary_open(raw_mask, config.opening_size) & valid
    brightness_change = np.mean(normalised_second - aligned.first, axis=0)
    regions, mask, region_count = _extract_regions(
        opened, magnitude, brightness_change, aligned, config
    )

    changed_pixels = int(mask.sum())
    changed_fraction = changed_pixels / valid_pixels if valid_pixels else 0.0

    metric = _crs_is_metric(aligned.crs)
    changed_area_m2 = (
        float(changed_pixels * aligned.pixel_area)
        if metric and aligned.pixel_area is not None
        else None
    )
    if aligned.georeferenced and not metric:
        warnings.append(
            "The images use a geographic (degree-based) CRS, so changed ground area in "
            "square metres is not reported; the percentage of the shared area is."
        )

    if changed_fraction > config.suspicious_changed_fraction:
        warnings.append(
            f"{changed_fraction:.0%} of the shared area is flagged as changed. A figure this "
            "large usually indicates a radiometric or registration problem rather than real "
            "change, so treat the result as unreliable."
        )

    band_deltas: list[float] = []
    if changed_pixels > 0:
        for band in range(aligned.band_count):
            delta = normalised_second[band][mask] - aligned.first[band][mask]
            band_deltas.append(float(delta.mean()))

    result = ChangeResult(
        algorithm_version=ALGORITHM_VERSION,
        method=method,
        config=config,
        registration=registration,
        magnitude=magnitude,
        mask=mask,
        valid=valid,
        threshold=threshold,
        threshold_method=threshold_method,
        otsu_threshold=otsu,
        noise_floor=noise_floor,
        changed_pixels=changed_pixels,
        valid_pixels=valid_pixels,
        changed_fraction=changed_fraction,
        raw_changed_pixels=raw_changed,
        regions=regions,
        region_count=region_count,
        brightness_delta=float(brightness_change[mask].mean()) if changed_pixels else 0.0,
        band_deltas=band_deltas,
        normalization=normalization,
        normalization_refined=refined,
        georeferenced=aligned.georeferenced,
        area_units="square metres" if metric else "pixels",
        changed_area_m2=changed_area_m2,
        warnings=warnings,
    )

    result.indices = _compute_indices(aligned, roles_first, roles_second, valid, mask)
    return result


# --- question interpretation -------------------------------------------------------
#
# Deciding *what a temporal question is asking for* is an inference concern, not a web one, so
# it lives beside the measurement. That also lets an evaluation harness route questions exactly
# the way the serving path does, without importing the API layer.


class ChangeQuestion(StrEnum):
    """What a temporal question is actually asking for."""

    #: "did anything change?", "is there evidence of change?"
    OCCURRED = "occurred"
    #: "what changed?", "describe the changes"
    WHAT = "what"
    #: "where did it change?", "which part changed?"
    WHERE = "where"
    #: "how much changed?", "how significant is the change?"
    MAGNITUDE = "magnitude"
    #: "did vegetation decrease?"
    VEGETATION = "vegetation"
    #: "did water coverage increase?"
    WATER = "water"
    #: "did built-up area increase?"
    BUILTUP = "builtup"
    #: "which image shows more vegetation?"
    WHICH_DATE = "which_date"
    #: Understood as temporal but not answerable from what is measured.
    UNSUPPORTED = "unsupported"


_WHERE_TERMS = ("where", "which part", "which area", "which region", "location", "located")
_MAGNITUDE_TERMS = (
    "how much",
    "how significant",
    "how large",
    "how big",
    "extent",
    "magnitude",
    "percentage",
    "how many",
    "area of change",
)
_OCCURRED_TERMS = (
    "did the area change",
    "did it change",
    "has it changed",
    "any change",
    "is there change",
    "evidence of change",
    "did anything change",
    "has the area changed",
)
_WHAT_TERMS = ("what changed", "what has changed", "describe", "what are the changes", "summar")
_WHICH_DATE_TERMS = ("which image", "which date", "which one", "more vegetation", "more water")

_VEGETATION_TERMS = ("vegetation", "vegetated", "forest", "green", "crop", "farmland", "tree")
_WATER_TERMS = ("water", "lake", "river", "flood", "reservoir", "wetland", "coast")
_BUILTUP_TERMS = (
    "urban",
    "built-up",
    "built up",
    "builtup",
    "building",
    "construction",
    "development",
    "settlement",
    "city",
    "infrastructure",
    "road",
)

_INCREASE_TERMS = (
    "increase", "increased", "grow", "grown", "growth", "expand", "expanded", "more",
    "gain", "rise", "risen", "higher",
)
_DECREASE_TERMS = (
    "decrease", "decreased", "reduce", "reduced", "shrink", "shrunk", "loss", "lost",
    "less", "fewer", "decline", "declined", "lower", "deforest",
)


def _hits(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def classify_change_question(query: str) -> tuple[ChangeQuestion, str | None]:
    """Decide what a temporal question asks, and its expected direction if any.

    Returns ``(kind, direction)`` where direction is ``"increase"``, ``"decrease"`` or None.
    Topic questions are checked before generic ones so "did vegetation decrease?" is treated
    as a vegetation question rather than a generic "did it change".
    """
    text = query.lower().strip()
    direction: str | None = None
    if _hits(text, _INCREASE_TERMS):
        direction = "increase"
    if _hits(text, _DECREASE_TERMS):
        direction = "decrease"

    if _hits(text, _WHICH_DATE_TERMS) and (
        _hits(text, _VEGETATION_TERMS) or _hits(text, _WATER_TERMS)
    ):
        return ChangeQuestion.WHICH_DATE, direction

    # Topic-specific questions first: they carry the most specific answerable requirement.
    if _hits(text, _VEGETATION_TERMS):
        return ChangeQuestion.VEGETATION, direction
    if _hits(text, _WATER_TERMS):
        return ChangeQuestion.WATER, direction
    if _hits(text, _BUILTUP_TERMS):
        return ChangeQuestion.BUILTUP, direction

    if _hits(text, _WHERE_TERMS):
        return ChangeQuestion.WHERE, direction
    if _hits(text, _MAGNITUDE_TERMS):
        return ChangeQuestion.MAGNITUDE, direction
    if _hits(text, _WHAT_TERMS):
        return ChangeQuestion.WHAT, direction
    if _hits(text, _OCCURRED_TERMS):
        return ChangeQuestion.OCCURRED, direction

    # Anything else that mentions change at all gets the general description.
    if re.search(r"\bchang|\bdiffer|\bcompar", text):
        return ChangeQuestion.WHAT, direction
    return ChangeQuestion.UNSUPPORTED, direction


def magnitude_band(change: ChangeResult) -> str:
    """Coarse extent label for a measured change. Thresholds are explicit, not hidden."""
    if not change.significant:
        return "none"
    if change.changed_fraction >= 0.10:
        return "substantial"
    if change.changed_fraction >= 0.02:
        return "moderate"
    return "limited"
