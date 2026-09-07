"""Change-detection engineering-validation harness.

Runs the **production** deterministic change detector (`ml.inference.detect_change` via
`geospatial.align_pair`) on synthetic bi-temporal pairs with known ground-truth masks and
computes binary segmentation metrics (IoU / precision / recall / F1) plus per-pair
change/no-change correctness.

This is engineering validation on synthetic data — NOT LEVIR-CD / OSCD / SECOND. Those are
real datasets reported separately; see ml.evaluation.datasets and docs/benchmarking.md. The
detector measures *radiometric* change, not semantic class, which the reports state plainly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from geospatial import align_pair, read_metadata, resolve_band_roles
from ml.evaluation.metrics import segmentation_metrics
from ml.evaluation.synthetic import build_change_detection_fixtures
from ml.inference.temporal import detect_change


def _detect_mask(t1_path: str, t2_path: str) -> tuple[np.ndarray, Any]:
    aligned = align_pair(t1_path, t2_path)
    change = detect_change(
        aligned,
        roles_first=resolve_band_roles(read_metadata(t1_path)),
        roles_second=resolve_band_roles(read_metadata(t2_path)),
    )
    return np.asarray(change.mask, dtype=bool), change


def run(out_dir: str | Path) -> dict[str, Any]:
    """Build fixtures, run production change detection, score against known masks."""
    out = Path(out_dir)
    scenarios = build_change_detection_fixtures(out / "fixtures")

    predictions: list[dict[str, Any]] = []
    tp = fp = fn = tn = 0
    ious: list[float] = []
    occurrence_correct = 0

    for scenario in scenarios:
        pred_mask, change = _detect_mask(scenario.t1_path, scenario.t2_path)
        seg = segmentation_metrics(pred_mask, scenario.changed_mask)
        # Occurrence: did the detector flag significant change, and does that match truth?
        predicted_changed = bool(change.significant) and not change.inconclusive
        occurrence_correct += int(predicted_changed == scenario.changed)

        # Aggregate confusion only over changed fixtures for IoU (no-change has empty ref).
        if scenario.changed:
            ious.append(seg.iou)
        tp += seg.true_positive
        fp += seg.false_positive
        fn += seg.false_negative
        tn += seg.true_negative

        predictions.append(
            {
                "id": scenario.key,
                "changed_truth": scenario.changed,
                "changed_fraction_truth": scenario.changed_fraction,
                "predicted_changed": predicted_changed,
                "predicted_fraction": round(float(change.changed_fraction), 5),
                "inconclusive": bool(change.inconclusive),
                "iou": seg.iou,
                "precision": seg.precision,
                "recall": seg.recall,
                "f1": seg.f1,
            }
        )

    union = tp + fp + fn
    micro_iou = round(tp / union, 4) if union else 1.0
    precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

    return {
        "benchmark": "change_detection_synthetic",
        "kind": "engineering-validation (synthetic known masks; not LEVIR-CD/OSCD/SECOND)",
        "detector": "cva-robust (deterministic radiometric change vector analysis)",
        "scenarios": len(scenarios),
        "metrics": {
            "micro_iou": micro_iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou_changed": round(float(np.mean(ious)), 4) if ious else 0.0,
            "occurrence_accuracy": round(occurrence_correct / len(scenarios), 4)
            if scenarios
            else 0.0,
            "pixels": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        },
        "predictions": predictions,
        "note": (
            "Detects radiometric change, not semantic class. no-change pairs verify "
            "specificity; block pairs verify localization against a known mask."
        ),
    }
