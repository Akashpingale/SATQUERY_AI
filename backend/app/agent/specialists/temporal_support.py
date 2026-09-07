"""Shared machinery behind the two Phase 6 temporal specialists.

Both change description and change VQA answer questions about the *same* measurement, so
the pair resolution, co-registration, detection, evidence, and optional semantic reading
live here once. The specialists differ only in how they turn the measurement into words.

Guard rails enforced in this module, each because getting it wrong would produce a
confident falsehood:

- A pair that cannot be placed on a common grid yields ``aligned is None`` and an
  explanation, never a comparison of mismatched pixels.
- Optical and SAR cannot be differenced against each other; that combination is refused.
- Every operation applied to the imagery is recorded for the execution trace.
- The land-cover reading of a changed region is optional and clearly labelled. When it is
  unavailable or weak, the cause is reported as undetermined rather than guessed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from geospatial import AlignedPair, AlignmentError, align_pair, read_metadata, resolve_band_roles
from geospatial.bands import BandRoles
from ml.inference import ChangeConfig, ChangeResult, ModelUnavailableError, detect_change
from ml.inference.change_map import crop_pair, render_change_map, render_comparison_strip

from ...core.logging import get_logger
from ...models_db.raster_asset import RasterAsset
from ...services import storage
from ..types import EvidenceItem, TraceEntry
from .base import SpecialistContext

logger = get_logger("satquery.temporal")

#: Change artifacts are written beside the Phase 4 overlays so one evidence endpoint serves
#: every rendered artifact.
EVIDENCE_SUBDIR = "overlays"

#: Minimum land-cover probability before a concept is used to describe what a region became.
#: Below this the reading is too weak to name a cause, and the result says so.
SEMANTIC_MIN_PROBABILITY = 0.20
#: Minimum probability shift before a land-cover change is described as a transition.
SEMANTIC_MIN_SHIFT = 0.10


def _artifact_path(token: str) -> str:
    return f"{storage.EVIDENCE_DIR}/{EVIDENCE_SUBDIR}/{token}.png"


def _reference(token: str) -> str:
    return f"/api/v1/evidence/{token}"


@dataclass
class SemanticReading:
    """Optional land-cover reading of a changed region, from real model scores."""

    available: bool
    reason: str = ""
    variant: str = ""
    model_version: str = ""
    before_label: str = ""
    before_probability: float = 0.0
    after_label: str = ""
    after_probability: float = 0.0
    #: Concepts whose probability moved most, as ``(label, delta)``.
    shifts: list[tuple[str, float]] = field(default_factory=list)

    @property
    def transition_supported(self) -> bool:
        """Whether the scores support naming a before -> after transition."""
        return (
            self.available
            and self.before_label != self.after_label
            and self.after_probability >= SEMANTIC_MIN_PROBABILITY
            and abs(self.after_probability - self.before_probability) >= SEMANTIC_MIN_SHIFT
        )

    def as_detail(self) -> dict[str, object]:
        return {
            "available": self.available,
            "reason": self.reason,
            "variant": self.variant,
            "model_version": self.model_version,
            "before_label": self.before_label,
            "before_probability": round(self.before_probability, 4),
            "after_label": self.after_label,
            "after_probability": round(self.after_probability, 4),
            "largest_shifts": [[label, round(delta, 4)] for label, delta in self.shifts[:5]],
            "transition_supported": self.transition_supported,
            "note": (
                "Zero-shot land-cover scoring of the changed region, before and after. "
                "Indicative only: no change-type classifier was trained or evaluated."
            ),
        }


@dataclass
class TemporalAnalysis:
    """Everything the temporal specialists need, measured once."""

    first: RasterAsset
    second: RasterAsset
    aligned: AlignedPair | None
    change: ChangeResult | None
    evidence: list[EvidenceItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace_notes: list[TraceEntry] = field(default_factory=list)
    semantic: SemanticReading | None = None
    #: Set when the pair could not be analysed at all; carries a user-facing explanation.
    blocked_reason: str = ""
    blocked_code: str = ""
    change_map_reference: str | None = None
    strip_reference: str | None = None

    @property
    def usable(self) -> bool:
        return self.aligned is not None and self.change is not None


def resolve_pair(
    context: SpecialistContext,
) -> tuple[RasterAsset | None, RasterAsset | None, list[str], str]:
    """Pick the earlier and later images.

    Explicit ``t1``/``t2`` roles are authoritative. When they are absent the upload order is
    used and that assumption is reported, because calling the wrong image "earlier" would
    invert the direction of every answer.

    Returns ``(first, second, warnings, order_source)``.
    """
    warnings: list[str] = []
    assets = context.assets()
    if len(assets) < 2:
        return None, None, ["Fewer than two readable images were supplied."], "none"

    by_id = {asset.id: asset for asset in assets}
    t1_image = context.inputs.by_role("t1")
    t2_image = context.inputs.by_role("t2")

    if t1_image is not None and t2_image is not None:
        first = by_id.get(t1_image.asset_id)
        second = by_id.get(t2_image.asset_id)
        if first is not None and second is not None:
            return first, second, warnings, "t1/t2 roles"

    warnings.append(
        "The uploads are not labelled T1/T2, so the first was treated as the earlier date "
        "and the second as the later one. Re-upload with T1/T2 roles if that is reversed."
    )
    return assets[0], assets[1], warnings, "upload order (roles not supplied)"


def _band_roles(asset: RasterAsset) -> BandRoles | None:
    """Band roles the raster itself declares, or None when it declares none."""
    try:
        metadata = read_metadata(storage.absolute_path(asset.storage_path))
    except Exception:  # noqa: BLE001 - metadata is optional enrichment here
        logger.warning("Band role lookup failed for asset %s", asset.id, exc_info=True)
        return None
    return resolve_band_roles(metadata)


def _semantic_reading(
    aligned: AlignedPair, change: ChangeResult
) -> SemanticReading:
    """Score the largest changed region's land cover before and after.

    This is the only learned component in the temporal path, and it is deliberately
    subordinate to the measurement: it can describe what a region looks like, and it never
    decides *whether* something changed. If the model is unavailable the change result stands
    on its own with the cause reported as undetermined.
    """
    if not change.regions:
        return SemanticReading(
            available=False, reason="no coherent changed region was large enough to inspect"
        )

    crops = crop_pair(aligned, change.regions[0].box)
    if crops is None:
        return SemanticReading(
            available=False,
            reason="the changed region is too small for a land-cover reading to be meaningful",
        )

    try:
        from ml.inference import clip_backend, resolve_variant

        variant, _note = resolve_variant(prefer_adapted=True)
        before_scores = clip_backend.score_land_cover(crops[0], variant=variant)
        after_scores = clip_backend.score_land_cover(crops[1], variant=variant)
    except ModelUnavailableError as exc:
        return SemanticReading(
            available=False,
            reason=f"the land-cover model is unavailable ({exc})",
        )
    except Exception:  # noqa: BLE001 - the measurement must survive an optional extra
        logger.warning("Semantic reading of a changed region failed", exc_info=True)
        return SemanticReading(
            available=False, reason="the land-cover reading could not be computed"
        )

    before_top = before_scores.top
    after_top = after_scores.top
    if before_top is None or after_top is None:
        return SemanticReading(available=False, reason="the land-cover model returned no scores")

    before_map = {score.key: score for score in before_scores.scores}
    shifts: list[tuple[str, float]] = []
    for score in after_scores.scores:
        previous = before_map.get(score.key)
        if previous is not None:
            shifts.append((score.label, score.probability - previous.probability))
    shifts.sort(key=lambda item: abs(item[1]), reverse=True)

    return SemanticReading(
        available=True,
        variant=str(variant),
        model_version=after_scores.version or after_scores.model_id,
        before_label=before_top.label,
        before_probability=float(before_top.probability),
        after_label=after_top.label,
        after_probability=float(after_top.probability),
        shifts=shifts,
    )


def _alignment_trace(aligned: AlignedPair) -> TraceEntry:
    """Record exactly what was done to the imagery before comparison."""
    operations = "; ".join(aligned.operations) if aligned.operations else "no changes required"
    scope = (
        f"{aligned.width}x{aligned.height} px, {aligned.band_count} band(s), "
        f"{aligned.valid_fraction:.0%} of the grid valid in both images"
    )
    detail = f"{operations}. Common grid: {scope}."
    if aligned.crs:
        detail += f" CRS {aligned.crs}."
    return TraceEntry(step="Co-registration", status="ok", detail=detail)


def _detection_trace(change: ChangeResult) -> TraceEntry:
    """Record the measurement, including the threshold that was actually used."""
    if change.inconclusive:
        return TraceEntry(
            step="Change analysis",
            status="warning",
            detail=f"No change measurement produced: {change.inconclusive_reason}",
        )
    detail = (
        f"{change.method.value} v{change.algorithm_version}; threshold {change.threshold:.4f} "
        f"chosen by {change.threshold_method} (Otsu {change.otsu_threshold:.4f}, noise floor "
        f"{change.noise_floor:.4f}); {change.changed_pixels} of {change.valid_pixels} valid px "
        f"changed ({change.changed_fraction:.2%}); {change.region_count} coherent region(s); "
        f"coherence {change.coherence:.2f}"
    )
    if change.normalization_refined:
        detail += "; radiometric fit refined on unchanged pixels"
    return TraceEntry(step="Change analysis", status="ok", detail=detail)


def _registration_trace(change: ChangeResult) -> TraceEntry:
    registration = change.registration
    status = "ok" if registration.aligned else "warning"
    return TraceEntry(
        step="Registration check",
        status=status,
        detail=(
            f"best correlation {registration.peak_correlation:.3f} at offset "
            f"{registration.peak_offset} (zero-offset {registration.zero_correlation:.3f}); "
            f"shift explains {registration.shift_explains:.0%} of residual difference"
        ),
    )


def analyze_pair(context: SpecialistContext) -> TemporalAnalysis:
    """Co-register the pair and measure change. The single entry point for both specialists."""
    first, second, warnings, order_source = resolve_pair(context)
    if first is None or second is None:
        return TemporalAnalysis(
            first=first,  # type: ignore[arg-type]
            second=second,  # type: ignore[arg-type]
            aligned=None,
            change=None,
            warnings=warnings,
            blocked_code="insufficient_inputs",
            blocked_reason=(
                "Temporal change analysis needs two readable images of the same area."
            ),
        )

    analysis = TemporalAnalysis(first=first, second=second, aligned=None, change=None)
    analysis.warnings.extend(warnings)
    analysis.trace_notes.append(
        TraceEntry(
            step="Temporal inputs",
            status="ok",
            detail=(
                f"earlier={first.original_filename}, later={second.original_filename} "
                f"(order from {order_source})"
            ),
        )
    )

    # Optical and SAR measure different physical quantities. Differencing one against the
    # other produces a number with no meaning, so the combination is refused outright.
    modalities = {first.modality, second.modality}
    if "optical" in modalities and "sar" in modalities:
        analysis.blocked_code = "modality_mismatch"
        analysis.blocked_reason = (
            "One image is optical and the other is SAR. They measure different physical "
            "quantities, so differencing them would not describe change. Supply two images "
            "of the same modality; joint optical-SAR analysis is a separate capability."
        )
        return analysis

    is_sar = modalities == {"sar"}

    try:
        aligned = align_pair(
            storage.absolute_path(first.storage_path),
            storage.absolute_path(second.storage_path),
        )
    except AlignmentError as exc:
        analysis.blocked_code = exc.code
        analysis.blocked_reason = exc.message
        analysis.trace_notes.append(
            TraceEntry(step="Co-registration", status="failed", detail=exc.message)
        )
        return analysis
    except Exception as exc:  # noqa: BLE001 - a bad raster must not crash the request
        logger.warning("Co-registration failed for %s/%s", first.id, second.id, exc_info=True)
        analysis.blocked_code = "alignment_failed"
        analysis.blocked_reason = (
            f"The image pair could not be prepared for comparison ({type(exc).__name__})."
        )
        analysis.trace_notes.append(
            TraceEntry(
                step="Co-registration", status="failed", detail=analysis.blocked_reason
            )
        )
        return analysis

    analysis.aligned = aligned
    analysis.trace_notes.append(_alignment_trace(aligned))

    change = detect_change(
        aligned,
        is_sar=is_sar,
        roles_first=_band_roles(first),
        roles_second=_band_roles(second),
        config=_config_from(context),
    )
    analysis.change = change
    analysis.trace_notes.append(_registration_trace(change))
    analysis.trace_notes.append(_detection_trace(change))
    analysis.warnings.extend(change.warnings)

    if not aligned.georeferenced:
        analysis.warnings.append(
            "Neither image carries a CRS or transform, so the changed areas are reported in "
            "pixels and cannot be tied to ground coordinates."
        )

    if change.inconclusive:
        return analysis

    analysis.semantic = _semantic_reading(aligned, change)
    analysis.evidence = build_evidence(analysis)
    return analysis


def _config_from(context: SpecialistContext) -> ChangeConfig:
    """Build the detector config from allowlisted plan parameters.

    Only parameters the registry declares can reach here; ``validate_parameters`` has
    already coerced and range-checked them.
    """
    defaults = ChangeConfig()
    return ChangeConfig(
        noise_sigma=float(context.parameters.get("noise_sigma", defaults.noise_sigma)),
        min_region_pixels=int(
            context.parameters.get("min_region_pixels", defaults.min_region_pixels)
        ),
        opening_size=defaults.opening_size,
        max_reported_regions=defaults.max_reported_regions,
        max_shift_pixels=defaults.max_shift_pixels,
        shift_search_max_side=defaults.shift_search_max_side,
        min_shift_explains=defaults.min_shift_explains,
        max_tolerable_shift_pixels=defaults.max_tolerable_shift_pixels,
        min_changed_fraction=float(
            context.parameters.get("min_changed_fraction", defaults.min_changed_fraction)
        ),
        suspicious_changed_fraction=defaults.suspicious_changed_fraction,
        min_valid_fraction=defaults.min_valid_fraction,
        low_correlation=defaults.low_correlation,
    )


def build_evidence(analysis: TemporalAnalysis) -> list[EvidenceItem]:
    """Render the change artifacts and assemble every evidence item.

    Rendering failures degrade to a warning: the measurement is the result, and an
    unrenderable overlay must not discard it.
    """
    aligned = analysis.aligned
    change = analysis.change
    if aligned is None or change is None:
        return []

    evidence: list[EvidenceItem] = []

    change_detail: dict[str, object] = {
        "method": change.method.value,
        "algorithm_version": change.algorithm_version,
        **change.stats_dict(),
        "thresholds": change.config.as_dict(),
        "normalization": change.normalization,
        "co_registration": aligned.operations,
        "grid": f"{aligned.width}x{aligned.height}",
        "crs": aligned.crs,
    }

    token = uuid.uuid4().hex[:16]
    try:
        render_change_map(aligned, change, storage.absolute_path(_artifact_path(token)))
        analysis.change_map_reference = _reference(token)
        evidence.append(
            EvidenceItem(
                kind="change_map",
                label="Measured change map (later image with flagged areas)",
                asset_id=analysis.second.id,
                detail=change_detail,
                reference=analysis.change_map_reference,
            )
        )
    except Exception:  # noqa: BLE001
        logger.warning("Change map rendering failed", exc_info=True)
        analysis.warnings.append("The change map could not be rendered for this result.")
        evidence.append(
            EvidenceItem(
                kind="comparison",
                label="Measured change statistics",
                detail=change_detail,
            )
        )

    strip_token = uuid.uuid4().hex[:16]
    try:
        render_comparison_strip(
            aligned, change, storage.absolute_path(_artifact_path(strip_token))
        )
        analysis.strip_reference = _reference(strip_token)
        evidence.append(
            EvidenceItem(
                kind="temporal_compare",
                label="T1 / T2 / change on the shared grid",
                detail={
                    "panels": ["T1 (earlier)", "T2 (later)", "measured change"],
                    "earlier": analysis.first.original_filename,
                    "later": analysis.second.original_filename,
                    "note": (
                        "All three panels share one grid, so the same position is the same "
                        "ground location in each."
                    ),
                },
                reference=analysis.strip_reference,
            )
        )
    except Exception:  # noqa: BLE001
        logger.warning("Comparison strip rendering failed", exc_info=True)

    for region in change.regions:
        detail = region.as_dict()
        detail["units"] = change.area_units
        # A distinct kind from the Phase 4 grounding `region`: the payload is different
        # (magnitude and brightness rather than detector scores), and reusing `region` would
        # make the UI render "coordinates unavailable" for a georeferenced result.
        evidence.append(
            EvidenceItem(
                kind="change_region",
                label=(
                    f"Changed region {region.index} - {region.pixel_count} px in the "
                    f"{region.position}"
                ),
                asset_id=analysis.second.id,
                detail=detail,
            )
        )

    available_indices = [item for item in change.indices if item.available]
    unavailable_indices = [item for item in change.indices if not item.available]
    if available_indices:
        evidence.append(
            EvidenceItem(
                kind="indices",
                label="Spectral index change",
                detail={
                    "indices": [item.as_dict() for item in available_indices],
                    "note": (
                        "Computed from bands the rasters explicitly declare, on unnormalised "
                        "reflectance."
                    ),
                },
            )
        )
    if unavailable_indices:
        evidence.append(
            EvidenceItem(
                kind="indices",
                label="Spectral indices not available",
                detail={
                    "indices": [item.as_dict() for item in unavailable_indices],
                    "note": (
                        "Indices are only computed when the raster declares which band is "
                        "which. Band order is not standardised, so guessing it would produce "
                        "a plausible but unfounded number."
                    ),
                },
            )
        )

    if analysis.semantic is not None:
        evidence.append(
            EvidenceItem(
                kind="scores",
                label="Land cover of the largest changed region, before and after",
                detail=analysis.semantic.as_detail(),
            )
        )

    evidence.append(
        EvidenceItem(
            kind="model",
            label="Method provenance",
            detail={
                "algorithm": change.method.value,
                "algorithm_version": change.algorithm_version,
                "learned_component": (
                    "none for detection; optional zero-shot CLIP reading for land-cover "
                    "description only"
                ),
                "device": "cpu",
                "deterministic": True,
                "note": (
                    "Change detection is algorithmic and reproducible: the same pair and "
                    "config always produce the same mask."
                ),
            },
        )
    )

    for asset in (analysis.first, analysis.second):
        if asset.preview_path:
            evidence.append(
                EvidenceItem(
                    kind="preview",
                    label=f"Preview of {asset.original_filename}",
                    asset_id=asset.id,
                    reference=f"/api/v1/uploads/{asset.id}/preview",
                )
            )

    return evidence


def measurement_score(change: ChangeResult) -> float | None:
    """Result strength for the confidence assessor, derived only from measured signals.

    Combines registration quality, spatial coherence, and how far the changed area sits
    from the significance floor. Returns ``None`` when nothing measurable was produced, so
    confidence becomes "Unable to determine" rather than a number with no basis.
    """
    if change.inconclusive:
        return None

    registration = change.registration
    # Correlation can be negative; clamp to a 0..1 quality reading.
    registration_quality = max(0.0, min(1.0, registration.peak_correlation))
    if not registration.aligned and registration.shift_explains >= change.config.min_shift_explains:
        registration_quality *= 0.6

    if not change.significant:
        # A confident "no significant change" is a real finding when the pair is well
        # registered and mostly valid. Its strength comes from registration quality.
        return round(0.55 * registration_quality + 0.25, 3)

    coherence = change.coherence
    # How decisively the changed fraction clears the significance floor, saturating at 10x.
    floor = max(change.config.min_changed_fraction, 1e-6)
    margin = min(1.0, (change.changed_fraction / floor - 1.0) / 9.0)
    # Magnitude headroom above the threshold, again measured rather than assumed.
    headroom = 0.0
    if change.changed_pixels > 0 and change.threshold > 0:
        mean_flagged = float(change.magnitude[change.mask].mean())
        excess = (mean_flagged - change.threshold) / max(change.threshold, 1e-6)
        headroom = min(1.0, max(0.0, excess))

    score = (
        0.35 * registration_quality
        + 0.30 * coherence
        + 0.20 * margin
        + 0.15 * headroom
    )
    if change.changed_fraction > change.config.suspicious_changed_fraction:
        score *= 0.5
    return round(max(0.0, min(1.0, score)), 3)
