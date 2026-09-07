"""Confidence assessment.

Confidence is **derived**, never invented. It combines only signals the system actually
has:

1. specialist score (how strong the underlying measurement/model output is),
2. routing score (how clearly the query mapped to the chosen task),
3. input compatibility (georeferencing present, pair overlap verified),
4. evidence presence (whether the answer is backed by anything inspectable),
5. warnings raised during execution.

Every returned assessment lists the signals that produced it, so a user or judge can see
why the level is what it is. When there is nothing to base a judgement on, the level is
``Unable to determine`` rather than a fabricated number.
"""

from __future__ import annotations

from .types import (
    ConfidenceAssessment,
    ConfidenceLevel,
    EvidenceItem,
    InputConfiguration,
    SpecialistResult,
)

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.45


def unable(reason: str) -> ConfidenceAssessment:
    """Confidence for a run that produced no analysable result."""
    return ConfidenceAssessment(level=ConfidenceLevel.UNABLE, score=None, signals=[reason])


def assess(
    result: SpecialistResult,
    routing_score: float,
    inputs: InputConfiguration,
    evidence: list[EvidenceItem],
) -> ConfidenceAssessment:
    """Combine available signals into a confidence level."""
    signals: list[str] = []

    # 1. Specialist score. Absent means we have no strength measure for the answer.
    if result.score is None:
        return unable("The specialist did not report a result strength.")
    specialist_score = max(0.0, min(1.0, result.score))
    signals.append(f"Specialist result strength {specialist_score:.2f}")

    # 2. Routing clarity.
    routing = max(0.0, min(1.0, routing_score))
    signals.append(f"Query routing score {routing:.2f}")

    # 3. Input quality: georeferencing is what makes geospatial claims verifiable.
    georeferenced = [image for image in inputs.images if image.is_georeferenced]
    if inputs.count == 0:
        input_quality = 0.0
        signals.append("No input imagery")
    elif len(georeferenced) == inputs.count:
        input_quality = 1.0
        signals.append("All inputs georeferenced")
    elif georeferenced:
        input_quality = 0.7
        signals.append("Some inputs lack georeferencing")
    else:
        input_quality = 0.5
        signals.append("No inputs georeferenced; geospatial claims unverifiable")

    # 4. Evidence backing.
    if evidence:
        evidence_quality = 1.0
        signals.append(f"{len(evidence)} evidence item(s) attached")
    else:
        evidence_quality = 0.4
        signals.append("No inspectable evidence attached")

    # Weighted blend: the specialist's own strength dominates, then input quality.
    score = (
        0.45 * specialist_score
        + 0.20 * routing
        + 0.20 * input_quality
        + 0.15 * evidence_quality
    )

    # 5. Warnings reduce confidence proportionally, floored so warnings alone cannot
    #    drive an otherwise solid result to zero.
    if result.warnings:
        penalty = min(0.30, 0.10 * len(result.warnings))
        score = max(0.0, score - penalty)
        signals.append(f"{len(result.warnings)} warning(s) reduced confidence")

    score = round(score, 3)

    if specialist_score == 0.0:
        # The specialist reported a definite negative (e.g. pair not compatible). That is
        # a reliable finding, not a weak one - but it is not an analytical answer either.
        level = ConfidenceLevel.HIGH if not result.warnings else ConfidenceLevel.MEDIUM
        signals.append("Specialist reported a definite negative finding")
    elif score >= HIGH_THRESHOLD:
        level = ConfidenceLevel.HIGH
    elif score >= MEDIUM_THRESHOLD:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    return ConfidenceAssessment(level=level, score=score, signals=signals)
