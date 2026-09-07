"""Confidence-calibration metrics for Phase 9 (CAL-1).

Pure functions. Given paired ``(confidence, correct)`` observations — a model confidence in
[0, 1] and whether that answer was actually correct — these compute whether the confidence is
*meaningful*: does 0.8 confidence really mean ~80% correct?

Nothing here decides thresholds or renames a score "calibrated". It only measures. Threshold
choices are a separate, documented policy step and must be driven by these measurements on a
validation split, never fitted on the data a number is later reported against.

Metrics:
- Reliability diagram bins (confidence bucket -> mean confidence, empirical accuracy, count).
- Expected / Maximum Calibration Error (ECE / MCE): gap between confidence and accuracy.
- Brier score: mean squared error of confidence against the 0/1 correctness outcome.
- Selective accuracy / coverage: accuracy over the answered subset at a confidence threshold.
- Abstention quality: how well "answer vs abstain" separates correct from incorrect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass
class CalibrationMetrics:
    samples: int = 0
    bins: list[ReliabilityBin] = field(default_factory=list)
    ece: float = 0.0
    mce: float = 0.0
    brier: float = 0.0
    overall_confidence: float = 0.0
    overall_accuracy: float = 0.0
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        return data


def _validate(confidences: list[float], correct: list[bool]) -> None:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    for c in confidences:
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"confidence out of [0,1]: {c}")


def reliability(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
    *,
    note: str = "",
) -> CalibrationMetrics:
    """Compute reliability bins, ECE, MCE and the Brier score.

    Uses equal-width bins over [0, 1]. Empty bins contribute nothing to ECE/MCE. With few
    samples the diagram is coarse; callers report ``samples`` so a small-``n`` result is not
    over-read.
    """
    _validate(confidences, correct)
    n = len(confidences)
    metrics = CalibrationMetrics(samples=n, note=note)
    if n == 0:
        return metrics

    edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # Last bin includes the right edge so confidence exactly 1.0 is counted.
        if b == n_bins - 1:
            idx = [i for i, c in enumerate(confidences) if lo <= c <= hi]
        else:
            idx = [i for i, c in enumerate(confidences) if lo <= c < hi]
        count = len(idx)
        if count == 0:
            metrics.bins.append(ReliabilityBin(round(lo, 3), round(hi, 3), 0, 0.0, 0.0))
            continue
        mean_conf = sum(confidences[i] for i in idx) / count
        acc = sum(1 for i in idx if correct[i]) / count
        gap = abs(mean_conf - acc)
        ece += (count / n) * gap
        mce = max(mce, gap)
        metrics.bins.append(
            ReliabilityBin(round(lo, 3), round(hi, 3), count, round(mean_conf, 4), round(acc, 4))
        )

    metrics.ece = round(ece, 4)
    metrics.mce = round(mce, 4)
    metrics.brier = round(
        sum((c - (1.0 if y else 0.0)) ** 2 for c, y in zip(confidences, correct, strict=True)) / n,
        4,
    )
    metrics.overall_confidence = round(sum(confidences) / n, 4)
    metrics.overall_accuracy = round(sum(1 for y in correct if y) / n, 4)
    return metrics


@dataclass
class SelectivePoint:
    threshold: float
    coverage: float
    selective_accuracy: float
    answered: int


def selective_accuracy(
    confidences: list[float],
    correct: list[bool],
    thresholds: list[float] | None = None,
) -> list[SelectivePoint]:
    """Accuracy over the subset whose confidence ≥ threshold, across thresholds.

    A well-behaved confidence yields higher selective accuracy as the threshold rises (you
    answer less, but more accurately). A flat or falling curve means the confidence does not
    rank correctness well.
    """
    _validate(confidences, correct)
    if thresholds is None:
        thresholds = [i / 10 for i in range(0, 10)]
    n = len(confidences)
    points: list[SelectivePoint] = []
    for t in thresholds:
        idx = [i for i, c in enumerate(confidences) if c >= t]
        answered = len(idx)
        acc = (sum(1 for i in idx if correct[i]) / answered) if answered else 0.0
        points.append(
            SelectivePoint(
                threshold=round(t, 3),
                coverage=round(answered / n, 4) if n else 0.0,
                selective_accuracy=round(acc, 4),
                answered=answered,
            )
        )
    return points


@dataclass
class AbstentionQuality:
    """How a hard answer/abstain decision separates correct from incorrect outcomes."""

    answered_correct: int = 0
    answered_incorrect: int = 0
    abstained_would_be_correct: int = 0
    abstained_would_be_incorrect: int = 0
    coverage: float = 0.0
    selective_accuracy: float = 0.0
    #: Fraction of abstentions that were justified (would have been wrong if answered).
    abstention_precision: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def abstention_quality(answered: list[bool], correct_if_answered: list[bool]) -> AbstentionQuality:
    """Evaluate answer/abstain decisions.

    Args:
        answered: Whether the system produced an answer for each sample.
        correct_if_answered: Whether the answer was (or, for abstentions, *would have been*)
            correct. For a genuine abstention this is the correctness of the answer the
            system withheld, so we can tell justified abstentions from over-caution.
    """
    if len(answered) != len(correct_if_answered):
        raise ValueError("inputs must be the same length")
    n = len(answered)
    ac = sum(1 for a, c in zip(answered, correct_if_answered, strict=True) if a and c)
    ai = sum(1 for a, c in zip(answered, correct_if_answered, strict=True) if a and not c)
    abc = sum(1 for a, c in zip(answered, correct_if_answered, strict=True) if not a and c)
    abi = sum(1 for a, c in zip(answered, correct_if_answered, strict=True) if not a and not c)
    answered_n = ac + ai
    abstained_n = abc + abi
    return AbstentionQuality(
        answered_correct=ac,
        answered_incorrect=ai,
        abstained_would_be_correct=abc,
        abstained_would_be_incorrect=abi,
        coverage=round(answered_n / n, 4) if n else 0.0,
        selective_accuracy=round(ac / answered_n, 4) if answered_n else 0.0,
        abstention_precision=round(abi / abstained_n, 4) if abstained_n else 0.0,
    )
