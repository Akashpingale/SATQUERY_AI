"""Phase 9 learned SECOND change-detection specialist.

Wraps the trained Siamese U-Net (`ml.inference.second_infer`) as a uniform specialist: a T1/T2
pair -> binary changed-pixel fraction + model provenance evidence. Registered, invocable, and
benchmarked (held-out IoU 0.42 / F1 0.59 vs a deterministic reference 0.22 / 0.37). It is NOT
the default change route: the deterministic Phase 6 CVA (georeferenced, general, robust) remains
the production default; this learned detector was trained on SECOND 0.1524 m aerial RGB and
detects binary change only. See docs/phase9_model_selection.md.
"""

from __future__ import annotations

from ...services import storage
from ..types import EvidenceItem, SpecialistResult
from .base import Specialist, SpecialistContext, SpecialistUnavailable


class SecondChangeSpecialist(Specialist):
    tool_id = "second_change_learned"

    def run(self, context: SpecialistContext) -> SpecialistResult:
        from ml.inference import second_infer

        status = second_infer.availability()
        if not status.available:
            raise SpecialistUnavailable(self.tool_id, status.reason)

        assets = context.assets()
        if len(assets) < 2:
            raise SpecialistUnavailable(
                self.tool_id, "The learned SECOND detector requires a T1/T2 image pair."
            )

        t1 = storage.absolute_path(assets[0].storage_path)
        t2 = storage.absolute_path(assets[1].storage_path)
        pred = second_infer.predict(str(t1), str(t2))

        percent = pred.changed_fraction * 100.0
        evidence = [
            EvidenceItem(
                kind="model",
                label="Learned SECOND change detector",
                asset_id=None,
                detail={
                    "model_id": "second_change_learned",
                    "version": pred.model_version,
                    "training_dataset": "SECOND train split (binary change, derived from labels)",
                    "held_out_iou": 0.4219,
                    "note": (
                        "Binary change (changed vs unchanged), not from-to semantic class. "
                        "Trained on 0.1524 m aerial RGB; confidence is an uncalibrated mean "
                        "probability over changed pixels."
                    ),
                },
                reference=None,
            ),
            EvidenceItem(
                kind="change_region",
                label="Learned change extent",
                asset_id=None,
                detail={
                    "changed_percent": round(percent, 2),
                    "mean_confidence": pred.mean_confidence,
                    "analysis_resolution_px": pred.size,
                },
                reference=None,
            ),
        ]
        return SpecialistResult(
            answer=(
                f"The learned SECOND detector estimates {percent:.1f}% of the scene changed "
                f"(binary change; mean confidence {pred.mean_confidence:.2f})."
            ),
            score=pred.mean_confidence,
            evidence=evidence,
            details={"changed_fraction": pred.changed_fraction},
        )
