"""Bi-temporal change specialists: change description and change VQA.

Both read one measurement produced by :mod:`temporal_support` and differ only in how they
put it into words.

The wording discipline is the point of this module. Every sentence either states a measured
quantity or explicitly declines to state one:

- Location comes from labelled connected components, so "in the northern part" is derived
  from a real centroid.
- Area comes from the pixel count times the common-grid pixel area, and is only given in
  square metres when the CRS is metric.
- Direction ("vegetation decreased") is only claimed from a spectral index that the raster's
  own band declarations made computable. Otherwise the answer says the measurement is not
  available and offers what *was* measured instead.
- Cause ("construction", "new buildings") is **never** claimed. The strongest statement
  available is a zero-shot land-cover reading of the region before and after, labelled as
  indicative.

A question the measurement cannot address is answered with a limitation, not a guess.
"""

from __future__ import annotations

from ml.inference import ChangeResult
from ml.inference.temporal import (
    ChangeQuestion,
    classify_change_question,
    magnitude_band,
)

from ..types import SpecialistResult
from .base import Specialist, SpecialistContext
from .temporal_support import TemporalAnalysis, analyze_pair, measurement_score

# Re-exported so callers and tests have one obvious import site for the temporal capability,
# while the classifier itself lives in the inference layer next to the measurement it routes to.
__all__ = [
    "ChangeQuestion",
    "ChangeVQASpecialist",
    "TemporalChangeSpecialist",
    "classify_change_question",
]


# --- shared wording helpers -------------------------------------------------------


def _extent_phrase(change: ChangeResult) -> str:
    """Measured extent of change, in the units the data actually supports."""
    percent = f"{change.changed_fraction * 100:.1f}% of the shared area"
    if change.changed_area_m2 is not None:
        if change.changed_area_m2 >= 1_000_000:
            return f"{percent} (about {change.changed_area_m2 / 1_000_000:.2f} km²)"
        return f"{percent} (about {change.changed_area_m2:,.0f} m²)"
    return f"{percent} ({change.changed_pixels:,} pixels)"


def _location_phrase(change: ChangeResult) -> str:
    """Where the change is, from region centroids."""
    if not change.regions:
        return ""
    largest = change.regions[0]
    phrase = f"the {largest.position}"
    if largest.centroid_lonlat:
        lon, lat = largest.centroid_lonlat
        phrase += f" (centred near {lat:.4f}, {lon:.4f})"
    if change.region_count > 1:
        phrase += f", with {change.region_count - 1} further separate area(s)"
    return phrase


def _brightness_phrase(change: ChangeResult) -> str:
    """Direction of the radiometric shift inside the changed areas."""
    if change.brightness_delta > 0:
        return "the changed areas are brighter in the later image"
    if change.brightness_delta < 0:
        return "the changed areas are darker in the later image"
    return "the changed areas show no net brightness shift"


def _cause_sentence(analysis: TemporalAnalysis) -> str:
    """What the changed area became - or an honest statement that it cannot be determined."""
    semantic = analysis.semantic
    undetermined = (
        " The available imagery does not provide sufficient evidence to determine the "
        "specific land-cover cause of the change."
    )
    if semantic is None or not semantic.available:
        return undetermined
    if semantic.transition_supported:
        return (
            f" A zero-shot land-cover reading of that area scores it as "
            f"'{semantic.before_label}' before ({semantic.before_probability:.0%}) and "
            f"'{semantic.after_label}' after ({semantic.after_probability:.0%}). That is "
            "indicative only - no change-type classifier was trained or evaluated, so the "
            "cause is not established."
        )
    return (
        f" A land-cover reading of that area returns '{semantic.after_label}' both before and "
        f"after ({semantic.after_probability:.0%}), so the imagery shows that something "
        "changed without supporting a specific land-cover cause."
    )


def _no_change_answer(analysis: TemporalAnalysis) -> str:
    change = analysis.change
    assert change is not None
    return (
        f"No significant change was detected between the two dates. After co-registration and "
        f"radiometric normalisation, {change.changed_fraction * 100:.2f}% of the shared area "
        f"exceeded the change threshold, below the "
        f"{change.config.min_changed_fraction * 100:.1f}% needed to report a coherent change, "
        f"and no changed region survived spatial-coherence filtering. The two images correlate "
        f"{change.registration.peak_correlation:.2f}."
    )


def _blocked_result(analysis: TemporalAnalysis) -> SpecialistResult:
    """Result for a pair that could not be analysed. No score, so confidence is 'unable'."""
    return SpecialistResult(
        answer=analysis.blocked_reason,
        score=None,
        evidence=analysis.evidence,
        details={"blocked": True, "code": analysis.blocked_code},
        warnings=analysis.warnings,
        trace_notes=analysis.trace_notes,
    )


def _inconclusive_result(analysis: TemporalAnalysis) -> SpecialistResult:
    change = analysis.change
    assert change is not None
    return SpecialistResult(
        answer=(
            "Unable to determine change from the available evidence. "
            f"{change.inconclusive_reason}"
        ),
        score=None,
        evidence=analysis.evidence,
        details={
            "inconclusive": True,
            "reason": change.inconclusive_reason,
            **change.stats_dict(),
        },
        warnings=analysis.warnings,
        trace_notes=analysis.trace_notes,
    )


def _base_details(analysis: TemporalAnalysis) -> dict[str, object]:
    change = analysis.change
    assert change is not None
    details: dict[str, object] = {
        "earlier": analysis.first.original_filename,
        "later": analysis.second.original_filename,
        **change.stats_dict(),
        "regions": [region.as_dict() for region in change.regions],
        "indices": [item.as_dict() for item in change.indices],
        "co_registration": analysis.aligned.operations if analysis.aligned else [],
    }
    if analysis.change_map_reference:
        details["change_map_ref"] = analysis.change_map_reference
    if analysis.strip_reference:
        details["comparison_ref"] = analysis.strip_reference
    if analysis.semantic is not None:
        details["semantic"] = analysis.semantic.as_detail()
    return details


class TemporalChangeSpecialist(Specialist):
    """Describes what changed between two dates, grounded in measured signals."""

    tool_id = "temporal_change"

    def run(self, context: SpecialistContext) -> SpecialistResult:
        analysis = analyze_pair(context)
        if analysis.blocked_reason:
            return _blocked_result(analysis)

        change = analysis.change
        assert change is not None
        if change.inconclusive:
            return _inconclusive_result(analysis)

        if not change.significant:
            return SpecialistResult(
                answer=_no_change_answer(analysis),
                score=measurement_score(change),
                evidence=analysis.evidence,
                details={**_base_details(analysis), "significant": False},
                warnings=analysis.warnings,
                trace_notes=analysis.trace_notes,
            )

        sentences = [
            f"Change was detected across {_extent_phrase(change)}, concentrated in "
            f"{_location_phrase(change)}."
        ]
        sentences.append(
            f"The flagged areas form {change.region_count} spatially coherent region(s) and "
            f"{_brightness_phrase(change)}."
        )

        ndvi = change.index_by_name("NDVI")
        if ndvi is not None and ndvi.available and ndvi.delta_in_changed is not None:
            direction = "fell" if ndvi.delta_in_changed < 0 else "rose"
            sentences.append(
                f"Within the changed areas the vegetation index (NDVI) {direction} by "
                f"{abs(ndvi.delta_in_changed):.3f}, measured from the red and near-infrared "
                f"bands the rasters declare."
            )

        ndwi = change.index_by_name("NDWI")
        if ndwi is not None and ndwi.available and ndwi.delta_in_changed is not None:
            direction = "fell" if ndwi.delta_in_changed < 0 else "rose"
            sentences.append(
                f"The water index (NDWI) {direction} by {abs(ndwi.delta_in_changed):.3f} in the "
                f"same areas."
            )

        answer = " ".join(sentences) + _cause_sentence(analysis)

        return SpecialistResult(
            answer=answer,
            score=measurement_score(change),
            evidence=analysis.evidence,
            details=_base_details(analysis),
            warnings=analysis.warnings,
            trace_notes=analysis.trace_notes,
        )


class ChangeVQASpecialist(Specialist):
    """Answers a specific question about the change between two dates."""

    tool_id = "change_vqa"

    def run(self, context: SpecialistContext) -> SpecialistResult:
        analysis = analyze_pair(context)
        if analysis.blocked_reason:
            return _blocked_result(analysis)

        change = analysis.change
        assert change is not None
        if change.inconclusive:
            return _inconclusive_result(analysis)

        kind, direction = classify_change_question(context.query)
        answer, extra_warnings, answered = self._answer(analysis, change, kind, direction)

        details = _base_details(analysis)
        details.update(
            {"question_kind": str(kind), "expected_direction": direction, "answered": answered}
        )

        return SpecialistResult(
            answer=answer,
            # An unanswerable question gets no strength, so confidence reports
            # "Unable to determine" rather than a number about a non-answer.
            score=measurement_score(change) if answered else None,
            evidence=analysis.evidence,
            details=details,
            warnings=analysis.warnings + extra_warnings,
            trace_notes=analysis.trace_notes,
        )

    # --- per-question answering ---------------------------------------------------

    def _answer(
        self,
        analysis: TemporalAnalysis,
        change: ChangeResult,
        kind: ChangeQuestion,
        direction: str | None,
    ) -> tuple[str, list[str], bool]:
        """Return ``(answer, extra_warnings, answered)``."""
        if kind is ChangeQuestion.UNSUPPORTED:
            return (
                (
                    "That question is not answerable from what this analysis measures. The "
                    "temporal comparison reports whether, where, and how much the imagery "
                    "changed, plus vegetation and water index shifts when the rasters declare "
                    "the required bands. Try asking what changed, where the change is, how "
                    "large it is, or whether vegetation or water changed."
                ),
                ["The question falls outside the measured temporal signals."],
                False,
            )

        if kind is ChangeQuestion.OCCURRED:
            if change.significant:
                return (
                    f"Yes. Change was measured across {_extent_phrase(change)}, forming "
                    f"{change.region_count} coherent region(s) in {_location_phrase(change)}."
                    + _cause_sentence(analysis),
                    [],
                    True,
                )
            return _no_change_answer(analysis), [], True

        if kind is ChangeQuestion.WHERE:
            if not change.significant:
                return (
                    "No coherent changed area was found to locate. "
                    + _no_change_answer(analysis),
                    [],
                    True,
                )
            parts = [f"The strongest change is in {_location_phrase(change)}."]
            for region in change.regions[:3]:
                area = (
                    f"{region.area_m2:,.0f} m²"
                    if region.area_m2 is not None
                    else f"{region.pixel_count:,} px"
                )
                where = f"the {region.position}"
                if region.centroid_lonlat:
                    lon, lat = region.centroid_lonlat
                    where += f" at {lat:.4f}, {lon:.4f}"
                parts.append(f"Region {region.index}: {where}, {area}.")
            return " ".join(parts), [], True

        if kind is ChangeQuestion.MAGNITUDE:
            if not change.significant:
                return _no_change_answer(analysis), [], True
            return (
                f"The change is {magnitude_band(change)} in extent: {_extent_phrase(change)} "
                f"exceeded the measured change threshold, spread over "
                f"{change.region_count} coherent "
                f"region(s). Spatial coherence of the flagged pixels is "
                f"{change.coherence:.0%}, and {_brightness_phrase(change)}.",
                [],
                True,
            )

        if kind is ChangeQuestion.WHAT:
            if not change.significant:
                return _no_change_answer(analysis), [], True
            return (
                f"Change affects {_extent_phrase(change)}, concentrated in "
                f"{_location_phrase(change)}, and {_brightness_phrase(change)}."
                + _cause_sentence(analysis),
                [],
                True,
            )

        if kind is ChangeQuestion.VEGETATION:
            return self._index_answer(
                analysis, change, "NDVI", "vegetation", direction, higher_means="more vegetation"
            )

        if kind is ChangeQuestion.WATER:
            return self._index_answer(
                analysis, change, "NDWI", "water", direction, higher_means="more open water"
            )

        if kind is ChangeQuestion.BUILTUP:
            return self._builtup_answer(analysis, change)

        if kind is ChangeQuestion.WHICH_DATE:
            return self._which_date_answer(analysis, change)

        return (
            "That temporal question could not be matched to a measured signal.",
            ["Unrecognised temporal question kind."],
            False,
        )

    def _index_answer(
        self,
        analysis: TemporalAnalysis,
        change: ChangeResult,
        index_name: str,
        topic: str,
        direction: str | None,
        *,
        higher_means: str,
    ) -> tuple[str, list[str], bool]:
        """Answer a direction question from a spectral index, or decline honestly."""
        index = change.index_by_name(index_name)
        if index is None or not index.available:
            reason = index.reason if index is not None else "the index was not computed"
            return (
                (
                    f"Whether {topic} increased or decreased cannot be determined from this "
                    f"pair: {reason}. A {index_name}-based answer needs the raster to declare "
                    f"which band is which, and inferring band order would produce a plausible "
                    f"but unfounded number. What was measured instead: "
                    f"{_extent_phrase(change) if change.significant else 'no significant change'}"
                    f"{', and ' + _brightness_phrase(change) if change.significant else ''}."
                    + _cause_sentence(analysis)
                ),
                [
                    f"{index_name} was not computable, so no {topic} trend is claimed. "
                    "Brightness change is reported instead and is not a vegetation or water "
                    "measurement."
                ],
                False,
            )

        delta = index.delta
        assert delta is not None
        moved = "increased" if delta > 0 else "decreased" if delta < 0 else "did not change"
        scene = (
            f"Across the shared area, {index_name} {moved} from {index.mean_before:.3f} to "
            f"{index.mean_after:.3f} (change {delta:+.3f}); higher {index_name} means "
            f"{higher_means}."
        )
        local = ""
        if index.delta_in_changed is not None and change.significant:
            local_moved = "lower" if index.delta_in_changed < 0 else "higher"
            local = (
                f" Within the areas flagged as changed it is {local_moved} by "
                f"{abs(index.delta_in_changed):.3f}, a larger shift than the scene average."
            )

        verdict = ""
        if direction is not None:
            expected_sign = 1.0 if direction == "increase" else -1.0
            agrees = delta * expected_sign > 0
            verdict = (
                f" That {'supports' if agrees else 'does not support'} a {direction} in {topic}."
            )

        warnings = [
            f"{index_name} is computed from declared bands and is uncalibrated; it indicates a "
            "relative trend, not an absolute area of gain or loss."
        ]
        return scene + local + verdict, warnings, True

    def _builtup_answer(
        self, analysis: TemporalAnalysis, change: ChangeResult
    ) -> tuple[str, list[str], bool]:
        """Built-up questions: report what is measured, refuse the classification claim.

        No built-up classifier was trained or evaluated, so answering "yes, urban area
        increased" would be a fabricated classification. The measured extent, brightness
        direction, and land-cover reading are reported instead, clearly labelled.
        """
        if not change.significant:
            return (
                "No significant change was measured, so there is no evidence of new "
                "construction or built-up expansion between these dates. "
                + _no_change_answer(analysis),
                [
                    "This system has no built-up/urban classifier; the statement is about "
                    "measured image change, not a land-use classification."
                ],
                True,
            )

        semantic = analysis.semantic
        reading = ""
        if semantic is not None and semantic.available:
            reading = (
                f" A zero-shot land-cover reading of the largest changed area scores it "
                f"'{semantic.before_label}' before and '{semantic.after_label}' after."
            )

        return (
            (
                f"Change was measured across {_extent_phrase(change)} in "
                f"{_location_phrase(change)}, and {_brightness_phrase(change)}.{reading} "
                "Whether that specifically represents new construction or built-up expansion "
                "cannot be established here: no built-up classifier was trained or evaluated, "
                "so the system reports the measured change and its land-cover reading rather "
                "than a land-use verdict."
            ),
            [
                "No built-up/urban classifier exists in this system. Construction is not "
                "confirmed or denied; only measured image change is reported."
            ],
            True,
        )

    def _which_date_answer(
        self, analysis: TemporalAnalysis, change: ChangeResult
    ) -> tuple[str, list[str], bool]:
        """Compare the two dates on whichever index is actually computable."""
        for name, topic in (("NDVI", "vegetation"), ("NDWI", "open water")):
            index = change.index_by_name(name)
            if index is not None and index.available and index.delta is not None:
                earlier = analysis.first.original_filename
                later = analysis.second.original_filename
                if index.delta > 0:
                    winner, value = later, index.mean_after
                    loser, other = earlier, index.mean_before
                elif index.delta < 0:
                    winner, value = earlier, index.mean_before
                    loser, other = later, index.mean_after
                else:
                    return (
                        f"The two dates show the same mean {name} ({index.mean_before:.3f}), so "
                        f"neither image has more {topic}.",
                        [],
                        True,
                    )
                return (
                    f"{winner} has more {topic}: its mean {name} is {value:.3f} against "
                    f"{other:.3f} for {loser}, measured over the shared area from the bands "
                    f"the rasters declare.",
                    [f"{name} is uncalibrated and indicates a relative comparison only."],
                    True,
                )

        return (
            (
                "Comparing the two dates for vegetation or water content needs a spectral "
                "index, and neither NDVI nor NDWI is computable here because the rasters do "
                "not declare which band is which. Guessing band order would produce a "
                "plausible but unfounded comparison, so no answer is given."
            ),
            ["No spectral index was computable for this pair."],
            False,
        )
