"""Shared machinery behind the optical+SAR cross-modal specialist.

Resolves which upload is optical and which is SAR, co-registers them onto one grid (reusing
the Phase 6 alignment), runs the measured cross-modal analysis, and assembles evidence. The
specialist itself only turns the measurement into words.

Guard rails, each because the alternative would fabricate:

- Exactly one optical and one SAR image are required; two-optical, two-SAR, or missing-modality
  pairs are refused with an explanation.
- A pair that cannot be co-registered yields a blocked result, never a comparison of mismatched
  pixels.
- SAR is described by measured backscatter/roughness only - never named land-cover classes.
- The optional land-cover reading is optical-only, zero-shot, and labelled indicative.
- Every co-registration operation is recorded for the trace.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from geospatial import AlignmentError, align_modalities, read_metadata, resolve_band_roles
from geospatial.bands import BandRoles
from ml.inference import ModelUnavailableError, analyze_cross_modal
from ml.inference.fusion import CrossModalResult
from ml.inference.fusion_render import (
    optical_pil,
    render_aligned_optical,
    render_aligned_sar,
    render_complementary_map,
    render_fusion_composite,
)

from ...core.logging import get_logger
from ...models_db.raster_asset import RasterAsset
from ...services import storage
from ..types import EvidenceItem, TraceEntry
from .base import SpecialistContext

logger = get_logger("satquery.fusion")

EVIDENCE_SUBDIR = "overlays"


def _artifact_path(token: str) -> str:
    return f"{storage.EVIDENCE_DIR}/{EVIDENCE_SUBDIR}/{token}.png"


def _reference(token: str) -> str:
    return f"/api/v1/evidence/{token}"


@dataclass
class FusionAnalysis:
    optical: RasterAsset | None
    sar: RasterAsset | None
    aligned: object | None = None
    result: CrossModalResult | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace_notes: list[TraceEntry] = field(default_factory=list)
    blocked_reason: str = ""
    blocked_code: str = ""

    @property
    def usable(self) -> bool:
        return self.aligned is not None and self.result is not None and not self.result.inconclusive


def resolve_modalities(
    context: SpecialistContext,
) -> tuple[RasterAsset | None, RasterAsset | None, str, str]:
    """Pick the optical and SAR assets. Returns ``(optical, sar, code, message)``.

    ``code`` is empty on success. Modality comes from the recorded asset modality; a pair that
    is not exactly one-optical + one-SAR is refused, because cross-modal fusion is undefined
    otherwise.
    """
    assets = context.assets()
    if len(assets) < 2:
        return None, None, "insufficient_inputs", "Cross-modal analysis needs two images."

    optical = [a for a in assets if a.modality == "optical"]
    sar = [a for a in assets if a.modality == "sar"]

    if len(optical) == 1 and len(sar) == 1:
        return optical[0], sar[0], "", ""

    if not sar:
        return None, None, "no_sar", (
            "Cross-modal optical-SAR analysis needs one SAR image, but none was supplied. "
            "Upload a SAR image in the SAR slot."
        )
    if not optical:
        return None, None, "no_optical", (
            "Cross-modal optical-SAR analysis needs one optical image, but none was supplied. "
            "Upload an optical image in the optical slot."
        )
    return None, None, "ambiguous_modalities", (
        f"Cross-modal analysis needs exactly one optical and one SAR image; received "
        f"{len(optical)} optical and {len(sar)} SAR. It cannot fuse two images of the same "
        "modality."
    )


def _band_roles(asset: RasterAsset) -> BandRoles | None:
    try:
        return resolve_band_roles(read_metadata(storage.absolute_path(asset.storage_path)))
    except Exception:  # noqa: BLE001
        logger.warning("Band role lookup failed for %s", asset.id, exc_info=True)
        return None


def _sar_tags(asset: RasterAsset) -> dict[str, str]:
    try:
        return read_metadata(storage.absolute_path(asset.storage_path)).tags
    except Exception:  # noqa: BLE001
        return {}


def _optical_landcover(aligned) -> dict[str, object] | None:
    """Optional zero-shot land-cover reading of the aligned optical image.

    Optical-only and clearly indicative: CLIP is a natural/RS optical model, so running it on
    SAR would be unfounded. Returns None when the model is unavailable.
    """
    try:
        from ml.inference import clip_backend, resolve_variant

        variant, _ = resolve_variant(prefer_adapted=True)
        scores = clip_backend.score_land_cover(optical_pil(aligned), variant=variant)
    except ModelUnavailableError:
        return None
    except Exception:  # noqa: BLE001 - an optional extra must not break the measurement
        logger.warning("Optical land-cover reading failed", exc_info=True)
        return None

    top = scores.top
    if top is None:
        return None
    return {
        "variant": str(variant),
        "model_version": scores.version or scores.model_id,
        "top_label": top.label,
        "top_probability": round(float(top.probability), 4),
        "ranked": [
            {"label": s.label, "probability": round(float(s.probability), 4)}
            for s in scores.scores[:5]
        ],
        "note": (
            "Zero-shot land-cover reading of the OPTICAL image only; indicative, not benchmark "
            "validated, and not applied to SAR."
        ),
    }


def _alignment_trace(aligned) -> TraceEntry:
    ops = "; ".join(aligned.operations) if aligned.operations else "no changes required"
    detail = (
        f"{ops}. Common grid: {aligned.width}x{aligned.height} px, "
        f"optical {aligned.optical_band_count} band(s) + SAR {aligned.sar_band_count} band(s), "
        f"{aligned.valid_fraction:.0%} valid in both."
    )
    if aligned.crs:
        detail += f" CRS {aligned.crs}."
    return TraceEntry(step="Co-registration", status="ok", detail=detail)


def analyze_fusion(context: SpecialistContext) -> FusionAnalysis:
    """Resolve the pair, co-register, and run the cross-modal measurement."""
    optical, sar, code, message = resolve_modalities(context)
    if code:
        return FusionAnalysis(
            optical=optical, sar=sar, blocked_code=code, blocked_reason=message,
            warnings=[message] if code == "insufficient_inputs" else [],
        )

    analysis = FusionAnalysis(optical=optical, sar=sar)
    analysis.trace_notes.append(
        TraceEntry(
            step="Modality resolution",
            status="ok",
            detail=f"optical={optical.original_filename}, SAR={sar.original_filename}",
        )
    )

    try:
        aligned = align_modalities(
            storage.absolute_path(optical.storage_path),
            storage.absolute_path(sar.storage_path),
        )
    except AlignmentError as exc:
        analysis.blocked_code = exc.code
        analysis.blocked_reason = exc.message
        analysis.trace_notes.append(
            TraceEntry(step="Co-registration", status="failed", detail=exc.message)
        )
        return analysis
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cross-modal alignment failed for %s/%s", optical.id, sar.id, exc_info=True)
        analysis.blocked_code = "alignment_failed"
        analysis.blocked_reason = (
            f"The optical/SAR pair could not be prepared for comparison ({type(exc).__name__})."
        )
        analysis.trace_notes.append(
            TraceEntry(step="Co-registration", status="failed", detail=analysis.blocked_reason)
        )
        return analysis

    analysis.aligned = aligned
    analysis.trace_notes.append(_alignment_trace(aligned))

    result = analyze_cross_modal(
        aligned, optical_roles=_band_roles(optical), sar_tags=_sar_tags(sar)
    )
    analysis.result = result
    analysis.warnings.extend(result.warnings)
    analysis.trace_notes.append(
        TraceEntry(
            step="Cross-modal analysis",
            status="ok" if not result.inconclusive else "warning",
            detail=(
                f"{result.algorithm_version}; SAR={result.sar_representation}; "
                + (
                    result.inconclusive_reason
                    if result.inconclusive
                    else (
                        f"correlation="
                        f"{'n/a' if result.correlation is None else round(result.correlation, 3)}, "
                        f"complementarity="
                        f"{'n/a' if result.complementarity is None else result.complementarity}, "
                        f"optical-only={result.optical_only_information:.2f} / "
                        f"SAR-only={result.sar_only_information:.2f} / "
                        f"shared={result.shared_information:.2f} structure"
                    )
                )
            ),
        )
    )

    if result.inconclusive:
        return analysis

    result.optical_landcover = _optical_landcover(aligned)
    analysis.evidence = build_evidence(analysis)
    return analysis


def build_evidence(analysis: FusionAnalysis) -> list[EvidenceItem]:
    aligned = analysis.aligned
    result = analysis.result
    if aligned is None or result is None:
        return []

    evidence: list[EvidenceItem] = []

    def render(kind_render, label: str, kind: str, detail: dict) -> None:
        token = uuid.uuid4().hex[:16]
        try:
            kind_render(aligned, storage.absolute_path(_artifact_path(token)))
            evidence.append(
                EvidenceItem(kind=kind, label=label, detail=detail, reference=_reference(token))
            )
        except Exception:  # noqa: BLE001 - a render failure must not discard the measurement
            logger.warning("Fusion render failed: %s", label, exc_info=True)
            analysis.warnings.append(f"The '{label}' visualization could not be rendered.")

    stats = result.stats_dict()

    render(
        render_fusion_composite,
        "Optical+SAR false-colour composite (R=SAR, G/B=optical)",
        "fusion_composite",
        {
            "channels": {
                "red": "SAR backscatter",
                "green": "optical brightness",
                "blue": "optical brightness",
            },
            "sar_representation": result.sar_representation,
            "correlation": stats["correlation"],
            "complementarity": stats["complementarity"],
            "note": (
                "Reddish = SAR-strong (rough/structured); cyan = optically bright, low SAR return."
            ),
        },
    )
    render(
        lambda a, d: render_complementary_map(a, result, d),
        "Complementary-information map",
        "fusion_complementary",
        {
            "optical_only_information": stats["optical_only_information"],
            "sar_only_information": stats["sar_only_information"],
            "shared_information": stats["shared_information"],
            "region_count": len(result.regions),
        },
    )
    render(
        render_aligned_optical,
        f"Aligned optical ({analysis.optical.original_filename})",
        "aligned_optical",
        {"bands": aligned.optical_band_count, "grid": f"{aligned.width}x{aligned.height}"},
    )
    render(
        render_aligned_sar,
        f"Aligned SAR ({analysis.sar.original_filename})",
        "aligned_sar",
        {
            "bands": aligned.sar_band_count,
            "representation": result.sar_representation,
            "grid": f"{aligned.width}x{aligned.height}",
        },
    )

    # Modality feature comparison (measured, side by side).
    evidence.append(
        EvidenceItem(
            kind="modality_comparison",
            label="Optical vs SAR measured features",
            detail={
                "optical": result.optical.as_dict(),
                "sar": result.sar.as_dict(),
                "correlation": stats["correlation"],
                "complementarity": stats["complementarity"],
                "sar_representation": result.sar_representation,
                "sar_calibrated": result.sar_calibrated,
                "note": (
                    "SAR features are measured backscatter/roughness statistics, not land-cover "
                    "classes."
                ),
            },
        )
    )

    # Cross-modal water agreement (the clearest complementary signal), when computable.
    evidence.append(
        EvidenceItem(
            kind="cross_modal_water",
            label="Optical vs SAR water agreement",
            detail=result.water.as_dict(),
        )
    )

    for region in result.regions:
        detail = region.as_dict()
        detail["units"] = result.area_units
        evidence.append(
            EvidenceItem(
                kind="complementary_region",
                label=(
                    f"Complementary region {region.index} ({region.kind.replace('_', ' ')}) - "
                    f"{region.pixel_count} px in the {region.position}"
                ),
                asset_id=analysis.sar.id if region.kind == "sar_structure" else analysis.optical.id,
                detail=detail,
            )
        )

    if result.optical_landcover is not None:
        evidence.append(
            EvidenceItem(
                kind="scores",
                label="Optical land-cover reading (indicative, optical only)",
                detail={
                    "concepts": [
                        {"key": c["label"], "label": c["label"], "probability": c["probability"]}
                        for c in result.optical_landcover["ranked"]  # type: ignore[index]
                    ],
                    "method": "zero-shot CLIP on optical",
                    "note": result.optical_landcover["note"],  # type: ignore[index]
                },
            )
        )

    evidence.append(
        EvidenceItem(
            kind="model",
            label="Method provenance",
            detail={
                "algorithm": "optical-SAR feature-level cross-modal analysis",
                "algorithm_version": result.algorithm_version,
                "learned_component": (
                    "none for fusion; optional zero-shot CLIP land-cover reading of the optical "
                    "image only"
                ),
                "device": "cpu",
                "deterministic": True,
                "sar_representation": result.sar_representation,
            },
        )
    )

    for asset in (analysis.optical, analysis.sar):
        if asset.preview_path:
            evidence.append(
                EvidenceItem(
                    kind="preview",
                    label=f"Original {asset.modality} ({asset.original_filename})",
                    asset_id=asset.id,
                    reference=f"/api/v1/uploads/{asset.id}/preview",
                )
            )

    return evidence


def measurement_score(result: CrossModalResult) -> float | None:
    """Evidence-derived strength for the confidence assessor.

    Combines overlap quality (how much of the scene is jointly valid), whether both modalities
    carry usable dynamic range (so the comparison is meaningful), and whether a concrete
    complementary signal was found. Returns None when inconclusive, so confidence becomes
    "Unable to determine" rather than a number with no basis. Never implies calibration.
    """
    if result.inconclusive:
        return None

    overlap_quality = max(0.0, min(1.0, result.valid_fraction))
    both_have_signal = 1.0 if (result.correlation is not None) else 0.5
    concrete_signal = 0.0
    if result.regions:
        concrete_signal = 1.0
    elif result.water.available and result.water.agree_fraction > 0:
        concrete_signal = 0.7
    elif result.complementarity is not None:
        concrete_signal = 0.5

    score = 0.4 * overlap_quality + 0.3 * both_have_signal + 0.3 * concrete_signal
    return round(max(0.0, min(1.0, score)), 3)
