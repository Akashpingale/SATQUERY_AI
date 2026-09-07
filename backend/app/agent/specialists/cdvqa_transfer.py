"""Phase 15 improved CDVQA specialist (clip_blip transfer model).

Wraps the newly TRAINED classifier at ``models/phase15/cdvqa_clip_blip/`` as a uniform
specialist. It answers a change-VQA question over the closed CDVQA vocabulary using image
features from two frozen encoders concatenated per image: CLIP-rsicd (512) + the external
``change_model`` BLIP vision encoder (768). It genuinely uses the already-trained
``change_model`` as a complementary representation source.

On the frozen CDVQA benchmark it improves over the Phase 9 baseline:
Test 0.7156 vs 0.7114, Test2 0.6668 vs 0.6632 (val 0.7200 vs 0.7166). It is the strongest
learned Change-VQA model in the project; still non-default (the deterministic detector stays
the general default), reachable via CHANGE_VQA_MODEL.
"""

from __future__ import annotations

from ..types import EvidenceItem, SpecialistResult
from .base import Specialist, SpecialistContext, SpecialistUnavailable
from .vision_support import load_asset_image, model_evidence, preview_evidence


class CDVQATransferSpecialist(Specialist):
    tool_id = "cdvqa_transfer"

    def run(self, context: SpecialistContext) -> SpecialistResult:
        from ml.inference import cdvqa_transfer_infer

        status = cdvqa_transfer_infer.availability()
        if not status.available:
            raise SpecialistUnavailable(self.tool_id, status.reason)

        assets = context.assets()
        if len(assets) < 2:
            raise SpecialistUnavailable(
                self.tool_id, "The improved CDVQA model requires a T1/T2 image pair."
            )

        question = context.query.strip() or "What has changed?"
        try:
            t1 = load_asset_image(assets[0]).image
            t2 = load_asset_image(assets[1]).image
            pred = cdvqa_transfer_infer.predict(t1, t2, question)
        except Exception as exc:  # noqa: BLE001
            return SpecialistResult(
                answer="The image pair could not be read for analysis.",
                warnings=[f"Raster read failed: {type(exc).__name__}."],
            )

        evidence: list[EvidenceItem] = [
            EvidenceItem(
                kind="model",
                label="Improved CDVQA model (Phase 15, CLIP+change_model BLIP features)",
                detail={
                    "model_id": "cdvqa_transfer",
                    "version": pred.model_version,
                    "checkpoint": "models/phase15/cdvqa_clip_blip/cdvqa_transfer_best.pt",
                    "device": pred.device,
                    "feature_sources": "CLIP-rsicd (512) + change_model BLIP vision (768)",
                    "training_dataset": "CDVQA (train split only) over SECOND imagery",
                    "benchmark": (
                        "CDVQA frozen Test 0.7156 / Test2 0.6668 "
                        "(vs Phase 9 baseline 0.7114 / 0.6632)"
                    ),
                    "note": (
                        "Closed 19-answer vocabulary. Uses the external change_model as a "
                        "complementary frozen feature source. Confidence is an uncalibrated "
                        "softmax."
                    ),
                },
            ),
            EvidenceItem(
                kind="scores",
                label="Answer probabilities",
                detail={
                    "concepts": [
                        {"key": ans, "label": ans, "probability": round(prob, 4)}
                        for ans, prob in pred.ranked
                    ],
                    "note": "Softmax over the closed CDVQA answer vocabulary (uncalibrated).",
                },
            ),
            *preview_evidence(assets[0]),
            model_evidence(
                "cdvqa_transfer",
                pred.device,
                "Trained CDVQA classifier on CLIP-rsicd + change_model BLIP features.",
            ),
        ]

        warnings = [
            "This answer comes from the Phase 15 improved CDVQA classifier (closed 19-answer "
            "vocabulary), trained on CDVQA/SECOND 0.1524 m aerial imagery; benefit on other "
            "sensors/resolutions is not established.",
            "Confidence is an uncalibrated softmax, not a probability of correctness.",
        ]

        return SpecialistResult(
            answer=(
                f"{pred.answer} (improved CDVQA model, confidence {pred.confidence:.2f})"
            ),
            score=pred.confidence,
            evidence=evidence,
            details={
                "model_id": "cdvqa_transfer",
                "answer_token": pred.answer,
                "confidence": pred.confidence,
            },
            warnings=warnings,
        )
