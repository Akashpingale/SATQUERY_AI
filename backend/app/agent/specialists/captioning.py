"""Remote-sensing scene description (captioning).

This is **retrieval-based**, not generative: the image is scored against a remote-sensing
concept vocabulary and the description is composed from concepts whose measured
probabilities clear a threshold. That is stated in the output and in the evidence, because
calling it "generated" would misrepresent how it works.

The trade-off is deliberate. A generative captioner trained on natural photographs produces
fluent but frequently wrong descriptions of satellite imagery; composing from a
remote-sensing-adapted model's actual scores yields a description that is plainer but
faithful to what the model measured.
"""

from __future__ import annotations

from ml.inference import ModelUnavailableError, clip_backend
from ml.inference.clip_backend import ScoreResult

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

#: Concepts at or above this probability are named as the scene's primary character.
PRIMARY_THRESHOLD = 0.15
#: Concepts at or above this probability are mentioned as secondary features.
SECONDARY_THRESHOLD = 0.05
MAX_SECONDARY = 3


def _compose(result: ScoreResult) -> tuple[str, list[str]]:
    """Compose a description from measured scores. Returns ``(text, mentioned_keys)``."""
    if not result.scores:  # pragma: no cover
        return "The image could not be characterised.", []

    top = result.scores[0]
    mentioned = [top.key]

    if top.probability >= PRIMARY_THRESHOLD:
        text = f"The scene appears to be dominated by {top.label} ({top.probability:.0%})"
    else:
        text = (
            f"No single feature dominates; the closest match is {top.label} "
            f"({top.probability:.0%})"
        )

    secondary = [
        score
        for score in result.scores[1 : 1 + MAX_SECONDARY + 1]
        if score.probability >= SECONDARY_THRESHOLD
    ][:MAX_SECONDARY]

    if secondary:
        listed = ", ".join(f"{s.label} ({s.probability:.0%})" for s in secondary)
        text += f", with signs of {listed}"
        mentioned.extend(s.key for s in secondary)

    return text + ".", mentioned


class CaptioningSpecialist(Specialist):
    """Produces a remote-sensing-aware description of a single image.

    Serves both ``captioning`` (Phase 4 baseline) and ``captioning_adapted`` (Phase 5 LoRA);
    the variant is fixed per instance.
    """

    def __init__(self, tool_id: str = "captioning", variant: str = "baseline") -> None:
        self.tool_id = tool_id
        self.variant = variant

    def run(self, context: SpecialistContext) -> SpecialistResult:
        asset = context.first_asset()
        if asset is None:
            return SpecialistResult(
                answer="No readable image was found for this request.",
                warnings=["The referenced upload could not be loaded."],
            )

        try:
            loaded = load_asset_image(asset)
        except Exception as exc:  # noqa: BLE001
            return SpecialistResult(
                answer="The image could not be read for analysis.",
                warnings=[f"Raster read failed: {type(exc).__name__}."],
            )

        try:
            result = clip_backend.score_concepts(loaded.image, variant=self.variant)
        except ModelUnavailableError:
            raise

        description, mentioned = _compose(result)
        warnings = baseline_warnings(asset) + adaptation_warnings(self.variant)
        warnings.append(
            "This description is composed from concept scores (retrieval-based), not "
            "generated free text."
        )

        top_probability = result.scores[0].probability if result.scores else 0.0
        if top_probability < PRIMARY_THRESHOLD:
            warnings.append("No concept scored strongly, so the description is tentative.")

        evidence: list[EvidenceItem] = [
            EvidenceItem(
                kind="scores",
                label="Concept probabilities behind the description",
                detail={
                    "model_id": result.model_id,
                    "mentioned": mentioned,
                    "concepts": [
                        {
                            "key": score.key,
                            "label": score.label,
                            "probability": round(score.probability, 4),
                        }
                        for score in result.scores[:8]
                    ],
                    "method": "retrieval-based composition from zero-shot concept scores",
                },
            ),
            model_evidence(
                result.model_id,
                result.device,
                "CLIP fine-tuned on RSICD; description composed from its concept scores."
                if self.variant != "adapted"
                else "CLIP fine-tuned on RSICD then LoRA-adapted on BigEarthNet.txt captions; "
                "description composed from its concept scores.",
            ),
            variant_evidence(self.variant, f"registry tool '{self.tool_id}'"),
            *preview_evidence(asset),
        ]

        return SpecialistResult(
            answer=description,
            score=float(top_probability),
            evidence=evidence,
            details={
                "method": "retrieval",
                "model_id": result.model_id,
                "model_variant": result.variant,
                "model_version": result.version,
                "mentioned_concepts": mentioned,
            },
            warnings=warnings,
        )
