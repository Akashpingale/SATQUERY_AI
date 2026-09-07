"""Metric primitives for Phase 9 evaluation.

Pure functions only: no model loading, no I/O, no randomness. Every metric here is a
standard, auditable formula so a reader can check a reported number by hand. Segmentation
metrics operate on boolean masks; classification metrics on normalised short strings.

These are shared by every benchmark harness so a metric is defined once and identically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


def normalize_text(text: str) -> str:
    """Reduce an answer to a comparable short form.

    Only whitespace, case, and trailing sentence punctuation are normalised. No synonym
    mapping is applied here: deciding that "increased" and "more" are equal is an evaluation
    policy choice that must live in a documented, benchmark-specific place, never hidden in a
    generic comparator where it would quietly change reported accuracy.
    """
    return " ".join(str(text).strip().lower().rstrip(".!?").split())


def exact_match(prediction: str, reference: str) -> bool:
    """True when the normalised prediction equals the normalised reference."""
    return normalize_text(prediction) == normalize_text(reference)


@dataclass
class ClassificationMetrics:
    """Accuracy with an explicit answered/abstained split.

    ``accuracy`` is over *all* samples, so declining a question counts against coverage, not
    as a correct or wrong answer. ``accuracy_on_answered`` is the accuracy over only the
    samples the system chose to answer. Reporting both prevents a system from looking good by
    answering only the easy questions.
    """

    samples: int = 0
    answered: int = 0
    correct: int = 0
    accuracy: float = 0.0
    accuracy_on_answered: float = 0.0
    coverage: float = 0.0
    per_group: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def classification_metrics(
    outcomes: list[tuple[bool, bool]],
    groups: list[str] | None = None,
) -> ClassificationMetrics:
    """Aggregate ``(answered, correct)`` outcomes into accuracy/coverage.

    Args:
        outcomes: One ``(answered, correct)`` pair per sample. ``correct`` must be False
            whenever ``answered`` is False.
        groups: Optional per-sample group label (e.g. question type) for a per-group
            breakdown.
    """
    total = len(outcomes)
    answered = sum(1 for a, _ in outcomes if a)
    correct = sum(1 for _, c in outcomes if c)
    metrics = ClassificationMetrics(
        samples=total,
        answered=answered,
        correct=correct,
        accuracy=round(correct / total, 4) if total else 0.0,
        accuracy_on_answered=round(correct / answered, 4) if answered else 0.0,
        coverage=round(answered / total, 4) if total else 0.0,
    )

    if groups is not None:
        if len(groups) != total:
            raise ValueError("groups must align 1:1 with outcomes")
        buckets: dict[str, list[tuple[bool, bool]]] = {}
        for label, outcome in zip(groups, outcomes, strict=True):
            buckets.setdefault(label, []).append(outcome)
        for label, entries in sorted(buckets.items()):
            n = len(entries)
            a = sum(1 for x, _ in entries if x)
            c = sum(1 for _, y in entries if y)
            metrics.per_group[label] = {
                "samples": float(n),
                "answered": float(a),
                "correct": float(c),
                "accuracy": round(c / n, 4) if n else 0.0,
                "coverage": round(a / n, 4) if n else 0.0,
            }
    return metrics


@dataclass
class SegmentationMetrics:
    """Binary (changed vs unchanged) segmentation metrics against a reference mask.

    Accuracy is reported but flagged as potentially misleading under class imbalance via
    ``positive_fraction`` (the share of reference pixels that are positive). IoU/precision/
    recall/F1 are the reliable figures for a sparse positive class.
    """

    pixels: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    positive_fraction: float = 0.0
    predicted_positive_fraction: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def segmentation_metrics(prediction: np.ndarray, reference: np.ndarray) -> SegmentationMetrics:
    """Compare a predicted boolean mask against a reference boolean mask.

    Both arrays must be the same shape. They are cast to bool, so a 0/1 integer mask works.
    """
    pred = np.asarray(prediction).astype(bool)
    ref = np.asarray(reference).astype(bool)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape} vs reference {ref.shape}")

    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    tn = int(np.logical_and(~pred, ~ref).sum())
    total = pred.size

    union = tp + fp + fn
    iou = tp / union if union else 1.0  # both empty -> perfect agreement
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return SegmentationMetrics(
        pixels=total,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        iou=round(iou, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round((tp + tn) / total, 4) if total else 0.0,
        positive_fraction=round((tp + fn) / total, 4) if total else 0.0,
        predicted_positive_fraction=round((tp + fp) / total, 4) if total else 0.0,
    )


def mean_and_ci(values: list[float], z: float = 1.96) -> dict[str, float]:
    """Mean, standard deviation, and a normal-approx confidence interval for a sample.

    The interval is the crude ``mean ± z·SE`` and is only meaningful for a reasonable sample
    size; callers report ``n`` alongside so a tiny-``n`` interval is not over-interpreted.
    """
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    arr = np.asarray(values, dtype="float64")
    n = int(arr.size)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(float(np.median(arr)), 4),
        "ci_low": round(mean - z * se, 4),
        "ci_high": round(mean + z * se, 4),
    }


def accuracy_ci(correct: int, total: int, z: float = 1.96) -> dict[str, float]:
    """Wilson score interval for a binomial accuracy, robust for small/edge proportions."""
    if total <= 0:
        return {"n": 0, "accuracy": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = (z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return {
        "n": total,
        "accuracy": round(p, 4),
        "ci_low": round(max(0.0, centre - half), 4),
        "ci_high": round(min(1.0, centre + half), 4),
    }
