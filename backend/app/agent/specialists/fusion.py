"""Optical + SAR cross-modal specialist.

One specialist serves the single ``optical_sar_fusion`` task, branching on the cross-modal
question. It never concatenates two separate captions: every answer is composed from the
measured cross-modal quantities (correlation/complementarity, the optical-only / SAR-only /
shared information partition, and cross-modal water agreement) that :mod:`fusion_support`
produced from the co-registered pair.

Wording discipline:

- SAR is described by measured backscatter and roughness, never named land-cover classes.
- "Which modality shows vegetation" is answered from NDVI when the optical bands are declared,
  and declined otherwise - SAR does not measure greenness, so no vegetation verdict is put on
  it.
- Complementarity, correlation, and agreement are stated as the measured numbers.
- A question the measurement cannot address returns a limitation, not a guess.
"""

from __future__ import annotations

from ml.inference import CrossModalResult
from ml.inference.fusion import CrossModalQuestion, classify_cross_modal_question

from ..types import SpecialistResult
from .base import Specialist, SpecialistContext
from .fusion_support import FusionAnalysis, analyze_fusion, measurement_score

__all__ = ["OpticalSARFusionSpecialist"]


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _complementarity_phrase(result: CrossModalResult) -> str:
    if result.complementarity is None:
        return (
            "A cross-modal correlation could not be computed because one modality is nearly "
            "uniform over the shared area"
        )
    strength = (
        "strongly complementary" if result.complementarity >= 0.6
        else "moderately complementary" if result.complementarity >= 0.3
        else "largely redundant"
    )
    return (
        f"The two modalities are {strength} here (correlation "
        f"{result.correlation:.2f}, complementarity {result.complementarity:.2f})"
    )


def _sar_structure_phrase(result: CrossModalResult) -> str:
    sar_only = result.sar_only_information
    if sar_only <= 0.01:
        return (
            "SAR adds little structural detail beyond what the optical texture already shows "
            f"here ({_pct(sar_only)} of the scene is structured in SAR but not optical)"
        )
    return (
        f"SAR shows structure the optical image does not over {_pct(sar_only)} of the scene "
        f"(rough/high-backscatter areas that are optically smooth)"
    )


def _optical_landcover_phrase(result: CrossModalResult) -> str:
    reading = result.optical_landcover
    if not reading:
        return ""
    return (
        f" An indicative optical land-cover reading suggests '{reading['top_label']}' "
        f"({_pct(float(reading['top_probability']))}); this is not benchmark validated and is "
        "not applied to the SAR image."
    )


def _region_phrase(result: CrossModalResult) -> str:
    if not result.regions:
        return ""
    parts = []
    for region in result.regions[:3]:
        where = f"the {region.position}"
        if region.centroid_lonlat:
            lon, lat = region.centroid_lonlat
            where += f" (near {lat:.4f}, {lon:.4f})"
        kind = (
            "SAR-revealed structure"
            if region.kind == "sar_structure"
            else "optical/SAR water agreement"
        )
        parts.append(f"{kind} in {where}")
    return " Notable regions: " + "; ".join(parts) + "."


class OpticalSARFusionSpecialist(Specialist):
    """Cross-modal optical+SAR analysis and VQA."""

    tool_id = "optical_sar_fusion"

    def run(self, context: SpecialistContext) -> SpecialistResult:
        analysis = analyze_fusion(context)
        if analysis.blocked_reason:
            return SpecialistResult(
                answer=analysis.blocked_reason,
                score=None,
                evidence=analysis.evidence,
                details={"blocked": True, "code": analysis.blocked_code},
                warnings=analysis.warnings,
                trace_notes=analysis.trace_notes,
            )

        result = analysis.result
        assert result is not None
        if result.inconclusive:
            return SpecialistResult(
                answer=(
                    "Unable to determine cross-modal information from the available evidence. "
                    f"{result.inconclusive_reason}"
                ),
                score=None,
                evidence=analysis.evidence,
                details={"inconclusive": True, "reason": result.inconclusive_reason},
                warnings=analysis.warnings,
                trace_notes=analysis.trace_notes,
            )

        kind, topic = classify_cross_modal_question(context.query)
        answer, extra_warnings, answered = self._answer(analysis, result, kind, topic)

        details = {
            "question_kind": str(kind),
            "topic": topic,
            "answered": answered,
            **result.stats_dict(),
            "regions": [region.as_dict() for region in result.regions],
        }
        return SpecialistResult(
            answer=answer,
            score=measurement_score(result) if answered else None,
            evidence=analysis.evidence,
            details=details,
            warnings=analysis.warnings + extra_warnings,
            trace_notes=analysis.trace_notes,
        )

    # --- per-question answering ---------------------------------------------------

    def _answer(
        self,
        analysis: FusionAnalysis,
        result: CrossModalResult,
        kind: CrossModalQuestion,
        topic: str | None,
    ) -> tuple[str, list[str], bool]:
        if kind is CrossModalQuestion.UNSUPPORTED:
            return (
                (
                    "That question is not answerable from what this cross-modal analysis "
                    "measures. It compares the co-registered optical and SAR images and can "
                    "report their complementary information, what SAR reveals beyond optical, "
                    "which modality carries a given feature's evidence, and whether they agree on "
                    "water. Try one of those."
                ),
                ["The question falls outside the measured cross-modal signals."],
                False,
            )

        if kind is CrossModalQuestion.WHICH_MODALITY:
            return self._which_modality(result, topic)

        if kind is CrossModalQuestion.AGREEMENT:
            return self._agreement(result), [], True

        if kind is CrossModalQuestion.SAR_REVEALS:
            return (
                f"{_sar_structure_phrase(result)}. SAR here is {result.sar_representation}; its "
                f"structured fraction is {_pct(result.sar.structured_fraction)} versus "
                f"{_pct(result.optical.structured_fraction)} for optical brightness, and "
                f"{_pct(float(result.sar.extra.get('dark_fraction', 0.0)))} of the scene is "
                f"SAR-dark (smooth surfaces such as calm water or roads).{_region_phrase(result)} "
                "SAR is illumination-independent, so this structural evidence does not depend on "
                "sunlight or cloud cover.",
                [],
                True,
            )

        if kind is CrossModalQuestion.COMPLEMENTARY:
            return (
                f"{_complementarity_phrase(result)}. Of the structured area, "
                f"{_pct(result.optical_only_information)} is distinct in optical only, "
                f"{_pct(result.sar_only_information)} in SAR only, and "
                f"{_pct(result.shared_information)} in both. "
                f"{_water_phrase(result)}{_optical_landcover_phrase(result)}"
                f"{_region_phrase(result)}",
                [],
                True,
            )

        # DESCRIBE (default for an optical+SAR pair)
        return (
            (
                f"Analysing the co-registered optical and SAR images together. "
                f"{_complementarity_phrase(result)}. "
                f"Optical brightness averages {result.optical.mean_level:.2f} with "
                f"{_pct(result.optical.structured_fraction)} structured; SAR "
                f"({result.sar_representation}) averages {result.sar.mean_level:.2f} with "
                f"{_pct(result.sar.structured_fraction)} structured. "
                f"{_sar_structure_phrase(result)}. {_water_phrase(result)}"
                f"{_optical_landcover_phrase(result)}{_region_phrase(result)}"
            ),
            [],
            True,
        )

    def _which_modality(
        self, result: CrossModalResult, topic: str | None
    ) -> tuple[str, list[str], bool]:
        if topic == "vegetation":
            ndvi = result.optical.extra.get("ndvi_mean")
            if ndvi is None:
                return (
                    "Which modality shows vegetation more strongly cannot be answered from a "
                    "direct measure here: NDVI is not computable because the optical bands are "
                    "not declared, and SAR backscatter does not measure greenness. Declare the "
                    "optical red/NIR bands to enable a vegetation comparison.",
                    ["NDVI not computable; no vegetation verdict placed on SAR."],
                    False,
                )
            return (
                f"Optical provides the more direct vegetation evidence: it yields a spectral "
                f"vegetation index (mean NDVI {ndvi:.3f}), which measures greenness directly. "
                f"SAR here ({result.sar_representation}) measures backscatter and roughness, "
                "which respond to structure and moisture rather than chlorophyll, so it does not "
                "provide a comparable direct vegetation measure.",
                [],
                True,
            )

        if topic == "water":
            answer = (
                self._agreement(result)
                + " Both modalities contribute complementary water evidence: optical via NDWI, "
                "SAR via smooth dark surfaces."
            )
            return answer, [], result.water.available

        if topic == "builtup":
            return (
                f"For built-up/structural evidence, SAR carries more here: its structured "
                f"fraction is {_pct(result.sar.structured_fraction)} versus "
                f"{_pct(result.optical.structured_fraction)} for optical brightness, and SAR "
                f"shows structure absent from optical over {_pct(result.sar_only_information)} of "
                "the scene. This is measured backscatter roughness, not a built-up classification "
                "- no built-up classifier was applied to either image.",
                ["Structural evidence is measured roughness, not a land-use classification."],
                True,
            )

        # No recognised topic: fall back to the general complementarity answer.
        return (
            f"{_complementarity_phrase(result)}. Optical carries spectral/reflectance evidence "
            f"and SAR carries structural/backscatter evidence; name a feature (vegetation, "
            f"water, or built structures) to compare which modality shows it more strongly.",
            [],
            True,
        )

    def _agreement(self, result: CrossModalResult) -> str:
        water = result.water
        if not water.available:
            base = (
                "Whether optical and SAR agree on water cannot be measured here: "
                f"{water.reason}."
            )
        else:
            base = (
                f"Optical (NDWI) and SAR (dark/smooth surfaces) agree on water over "
                f"{_pct(water.agree_fraction)} of the scene (IoU {water.agreement_iou:.2f}). "
                f"Optical flags {_pct(water.optical_water_fraction)} as water and SAR "
                f"{_pct(water.sar_dark_fraction)} as smooth/dark; "
                f"{_pct(water.sar_only_fraction)} is SAR-dark without optical water evidence "
                "(e.g. roads, shadows, or radar-smooth surfaces)."
            )
        if result.correlation is not None:
            base += (
                f" Overall the modalities correlate {result.correlation:.2f} "
                f"(complementarity {result.complementarity:.2f})."
            )
        return base


def _water_phrase(result: CrossModalResult) -> str:
    water = result.water
    if not water.available:
        return "Optical/SAR water agreement is not available (NDWI not computable)."
    return (
        f"Optical and SAR agree on water over {_pct(water.agree_fraction)} of the scene "
        f"(IoU {water.agreement_iou:.2f})."
    )
