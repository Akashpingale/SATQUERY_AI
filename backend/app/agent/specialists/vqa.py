"""Single-image Visual Question Answering.

Backed by CLIP fine-tuned on RSICD (``flax-community/clip-rsicd-v2``), applied zero-shot:
the image is compared against a remote-sensing concept vocabulary and the resulting
probabilities are turned into an answer.

Honesty rules enforced here:

- **Counting questions are refused.** A contrastive image-text model has no counting
  mechanism, so returning a number would be fabrication.
- **Every answer carries its measured probability**, and the score fed to the confidence
  layer is the model's own value - never a constant.
- **Uncertainty is stated.** When the winning concept's probability is weak, the answer says
  so instead of asserting a conclusion.
"""

from __future__ import annotations

from ml.inference import ModelUnavailableError, QuestionKind, analyze_question, clip_backend
from ml.inference.clip_backend import PRESENCE_THRESHOLD, ScoreResult

from ..types import EvidenceItem, SpecialistResult
from .base import Specialist, SpecialistContext
from .vision_support import (
    adaptation_warnings,
    baseline_warnings,
    load_asset_image,
    model_evidence,
    preview_evidence,
    variant_evidence,
)

#: Probability below which a top-ranked concept is reported as uncertain.
WEAK_EVIDENCE_THRESHOLD = 0.20
#: How many concepts to list when answering an open "what is visible" question.
OPEN_ANSWER_TOP_K = 4


def _scores_evidence(result: ScoreResult, limit: int = 8) -> EvidenceItem:
    """Machine-readable concept scores, so an answer can always be checked."""
    return EvidenceItem(
        kind="scores",
        label="Concept probabilities from the model",
        detail={
            "model_id": result.model_id,
            "concepts": [
                {"key": score.key, "label": score.label, "probability": round(score.probability, 4)}
                for score in result.scores[:limit]
            ],
            "note": (
                "Softmax over the model's concept vocabulary; values are relative to that "
                "vocabulary and are not calibrated probabilities of ground truth."
            ),
        },
    )


class VQASpecialist(Specialist):
    """Answers a natural-language question about a single image.

    One class serves both registry entries: ``vqa`` (Phase 4 baseline) and ``vqa_adapted``
    (Phase 5 LoRA model). The variant is fixed per instance so the registry, the plan, and
    the trace all name exactly which model ran.
    """

    def __init__(self, tool_id: str = "vqa", variant: str = "baseline") -> None:
        self.tool_id = tool_id
        self.variant = variant

    def run(self, context: SpecialistContext) -> SpecialistResult:
        asset = context.first_asset()
        if asset is None:
            return SpecialistResult(
                answer="No readable image was found for this request.",
                warnings=["The referenced upload could not be loaded."],
            )

        analysis = analyze_question(context.query)

        # Refuse counting rather than invent a number.
        if analysis.kind == QuestionKind.COUNT:
            return SpecialistResult(
                answer=(
                    "This question asks for a count, which the current model cannot provide. "
                    f"{analysis.reason} Ask whether a feature is present, or use a grounding "
                    "query to locate individual instances."
                ),
                score=None,
                details={"question_kind": str(analysis.kind), "refused": True},
                warnings=["Counting is outside this model's capability."],
            )

        try:
            loaded = load_asset_image(asset)
        except Exception as exc:  # noqa: BLE001 - surfaced as a specialist warning
            return SpecialistResult(
                answer="The image could not be read for analysis.",
                warnings=[f"Raster read failed: {type(exc).__name__}."],
            )

        try:
            if analysis.kind == QuestionKind.DOMINANT:
                result = clip_backend.score_land_cover(loaded.image, variant=self.variant)
            else:
                result = clip_backend.score_concepts(loaded.image, variant=self.variant)
        except ModelUnavailableError as exc:
            # Propagated so the orchestrator reports unavailability rather than guessing.
            raise exc

        method = (
            "CLIP fine-tuned on RSICD then LoRA-adapted on BigEarthNet.txt captions."
            if self.variant == "adapted"
            else "CLIP fine-tuned on RSICD, applied zero-shot with a prompt-template ensemble."
        )
        evidence: list[EvidenceItem] = [
            _scores_evidence(result),
            model_evidence(result.model_id, result.device, method),
            variant_evidence(self.variant, f"registry tool '{self.tool_id}'"),
            *preview_evidence(asset),
        ]
        warnings = baseline_warnings(asset) + adaptation_warnings(self.variant)

        if analysis.kind == QuestionKind.PRESENCE and analysis.concepts:
            answer, score = self._answer_presence(analysis, result, warnings)
        elif analysis.kind == QuestionKind.DOMINANT:
            answer, score = self._answer_dominant(result, warnings)
        else:
            answer, score = self._answer_open(result, warnings)

        return SpecialistResult(
            answer=answer,
            score=score,
            evidence=evidence,
            details={
                "question_kind": str(analysis.kind),
                "model_id": result.model_id,
                "model_variant": result.variant,
                "model_version": result.version,
                "top_concept": result.top.key if result.top else None,
                "top_probability": round(result.top.probability, 4) if result.top else None,
            },
            warnings=warnings,
        )

    # --- answer strategies --------------------------------------------------------

    def _answer_presence(
        self, analysis, result: ScoreResult, warnings: list[str]
    ) -> tuple[str, float]:
        """Answer a yes/no presence question using the concept's measured probability."""
        target = analysis.concepts[0]
        score = result.by_key(target.key)
        if score is None:  # pragma: no cover - target always in the vocabulary
            return (
                f"The model has no concept for '{target.key}', so presence cannot be assessed.",
                0.0,
            )

        rank = [s.key for s in result.scores].index(target.key) + 1
        present = score.probability >= PRESENCE_THRESHOLD or rank == 1

        if present:
            qualifier = "likely" if score.probability < WEAK_EVIDENCE_THRESHOLD else "probably"
            answer = (
                f"Yes - {score.label} is {qualifier} present. It scores "
                f"{score.probability:.0%} against the model's concept vocabulary "
                f"(rank {rank} of {len(result.scores)})."
            )
            if score.probability < WEAK_EVIDENCE_THRESHOLD:
                warnings.append(
                    f"The score for {score.label} is weak; treat this as a tentative reading."
                )
        else:
            leader = result.top
            answer = (
                f"No strong evidence of {score.label}. It scores only "
                f"{score.probability:.0%} (rank {rank} of {len(result.scores)})"
            )
            if leader:
                answer += f", while {leader.label} scores highest at {leader.probability:.0%}."
            else:
                answer += "."

        return answer, float(score.probability)

    def _answer_dominant(self, result: ScoreResult, warnings: list[str]) -> tuple[str, float]:
        """Answer 'what dominates' with the top-scoring land-cover concept."""
        top = result.top
        if top is None:  # pragma: no cover
            return "No land-cover concepts could be scored for this image.", 0.0

        runner_up = result.scores[1] if len(result.scores) > 1 else None
        answer = f"The scene is most consistent with {top.label} ({top.probability:.0%})."
        if runner_up:
            answer += (
                f" The next closest match is {runner_up.label} "
                f"({runner_up.probability:.0%})."
            )

        if top.probability < WEAK_EVIDENCE_THRESHOLD:
            answer += " No single land-cover class stands out clearly, so this is a weak reading."
            warnings.append("No land-cover class scored strongly; the scene may be mixed.")
        elif runner_up and (top.probability - runner_up.probability) < 0.05:
            answer += " The top two classes score closely, so the scene is likely mixed."
            warnings.append(
                "The top two land-cover classes are close; the answer is not clear-cut."
            )

        return answer, float(top.probability)

    def _answer_open(self, result: ScoreResult, warnings: list[str]) -> tuple[str, float]:
        """Answer an open question by listing the most likely scene content."""
        top = result.top
        if top is None:  # pragma: no cover
            return "Nothing could be scored for this image.", 0.0

        notable = [s for s in result.scores[:OPEN_ANSWER_TOP_K] if s.probability >= 0.05]
        if not notable:
            notable = result.scores[:1]

        listed = ", ".join(f"{s.label} ({s.probability:.0%})" for s in notable)
        answer = f"The most likely scene content is: {listed}."

        if top.probability < WEAK_EVIDENCE_THRESHOLD:
            warnings.append("No concept scored strongly; the scene content is uncertain.")

        return answer, float(top.probability)
