"""Phase 9 trained CDVQA specialist.

Wraps the learned CDVQA classifier (`ml.inference.cdvqa_infer`) as a uniform specialist:
T1 + T2 + question -> closed-vocabulary answer with a softmax confidence, plus model
provenance evidence.

This specialist is REGISTERED and invocable, and it is benchmarked (CDVQA held-out Test
0.71 / Test2 0.66). It is deliberately NOT the default route for change-VQA: the model was
trained on SECOND 0.1524 m aerial imagery with a closed 19-answer vocabulary, so the general
deterministic detector (`change_vqa`) remains the production default for arbitrary imagery.
See docs/phase9_model_selection.md.
"""

from __future__ import annotations

from ...services import storage
from ..types import EvidenceItem, SpecialistResult
from .base import Specialist, SpecialistContext, SpecialistUnavailable


class CDVQATrainedSpecialist(Specialist):
    tool_id = "cdvqa_trained"

    def run(self, context: SpecialistContext) -> SpecialistResult:
        from ml.inference import cdvqa_infer

        status = cdvqa_infer.availability()
        if not status.available:
            raise SpecialistUnavailable(self.tool_id, status.reason)

        assets = context.assets()
        if len(assets) < 2:
            raise SpecialistUnavailable(
                self.tool_id, "The trained CDVQA model requires a T1/T2 image pair."
            )

        t1 = storage.absolute_path(assets[0].storage_path)
        t2 = storage.absolute_path(assets[1].storage_path)
        prediction = cdvqa_infer.predict(str(t1), str(t2), context.query)

        evidence = [
            EvidenceItem(
                kind="model",
                label="Trained CDVQA model",
                asset_id=None,
                detail={
                    "model_id": "cdvqa_trained",
                    "version": prediction.model_version,
                    "training_dataset": "CDVQA (questions) + SECOND (imagery)",
                    "note": (
                        "Closed-vocabulary classifier; trained on 0.1524 m aerial imagery. "
                        "Confidence is a softmax probability, not calibrated."
                    ),
                },
                reference=None,
            ),
            EvidenceItem(
                kind="scores",
                label="Answer probabilities",
                asset_id=None,
                detail={
                    "concepts": [
                        {"key": ans, "label": ans, "probability": prob}
                        for ans, prob in prediction.ranked
                    ],
                    "note": "Softmax over the closed CDVQA answer vocabulary (uncalibrated).",
                },
                reference=None,
            ),
        ]
        return SpecialistResult(
            answer=(
                f"{prediction.answer} (trained CDVQA model, "
                f"confidence {prediction.confidence:.2f})"
            ),
            score=prediction.confidence,
            evidence=evidence,
            details={"answer_token": prediction.answer, "confidence": prediction.confidence},
        )
