"""Benchmark-ready interface for change-based VQA evaluation.

Purpose: make the Phase 6 temporal pipeline *callable by a benchmark harness*, so a CDVQA-style
evaluation can be run in Phase 9 without reworking the serving code. This module contains the
adapter and the metrics. It does **not** contain results.

**No CDVQA numbers are reported anywhere in this repository.** The dataset has not been
downloaded and this harness has not been run against it. Any accuracy figure would be
fabricated.

**Schema honesty.** The concrete field names and question-type taxonomy of the published CDVQA
distribution have **not been verified** here, because the dataset was not obtained. The
:class:`ChangeVQASample` schema below is therefore this project's own documented contract, and
converting a real CDVQA release into it is a small, explicit adapter step
(:func:`load_samples` reads the documented JSONL). That boundary is deliberate: it keeps an
unverified assumption out of the evaluation code, where it could silently corrupt a metric.

**Coverage is reported alongside accuracy, always.** A system that declines most questions and
answers a handful correctly would otherwise look excellent. :class:`ChangeVQAMetrics` therefore
carries ``answered`` and ``coverage`` next to ``accuracy``, and also reports accuracy over the
answered subset separately from accuracy over the whole set.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Returned when the pipeline cannot address a question from what it measures. Kept as an
#: explicit token rather than an empty string so a harness can distinguish "declined" from
#: "answered incorrectly" - they mean very different things about a system.
UNANSWERABLE = "unanswerable"


@dataclass(frozen=True)
class ChangeVQASample:
    """One bi-temporal question-answer triplet.

    Args:
        sample_id: Stable identifier used to join predictions back to references.
        before_path: Raster for the earlier date.
        after_path: Raster for the later date.
        question: Natural-language question about the change.
        answer: Reference answer, lowercased short form (e.g. ``"yes"``, ``"no"``).
        question_type: Optional grouping label from the source dataset, used for per-type
            accuracy. Passed through untouched; this module never invents one.
    """

    sample_id: str
    before_path: str
    after_path: str
    question: str
    answer: str
    question_type: str = "unspecified"


@dataclass
class ChangeVQAPrediction:
    """What the pipeline produced for one sample."""

    sample_id: str
    predicted: str
    #: False when the pipeline declined; ``predicted`` is then :data:`UNANSWERABLE`.
    answered: bool
    #: The internal question kind the query was routed to, for error analysis.
    question_kind: str = ""
    #: Measured change fraction, so a harness can correlate errors with change magnitude.
    changed_fraction: float | None = None
    #: Set when the pair could not be compared at all (misregistered, no overlap, ...).
    blocked_reason: str = ""
    #: Full prose answer, kept for qualitative review.
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChangeVQAMetrics:
    """Measured evaluation metrics. Never populated with estimates."""

    samples: int = 0
    answered: int = 0
    correct: int = 0
    #: Correct over *all* samples: declining counts against you.
    accuracy: float = 0.0
    #: Correct over the answered subset only.
    accuracy_on_answered: float = 0.0
    #: Share of samples the pipeline was willing to answer.
    coverage: float = 0.0
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Free-text note recording what this run actually was.
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_samples(path: str | Path) -> list[ChangeVQASample]:
    """Load samples from a JSONL file matching :class:`ChangeVQASample`.

    One JSON object per line with keys ``sample_id``, ``before_path``, ``after_path``,
    ``question``, ``answer``, and optionally ``question_type``.

    Raises:
        FileNotFoundError: if the file is absent. Callers must report the dataset as
            unavailable rather than substituting anything.
        ValueError: if a line is missing a required key, so a malformed conversion fails
            loudly instead of silently evaluating on partial data.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Change-VQA samples not found at {path}. The dataset must be obtained and "
            "converted to the documented JSONL schema before evaluation."
        )

    required = ("sample_id", "before_path", "after_path", "question", "answer")
    samples: list[ChangeVQASample] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"{path}:{number} is missing required key(s): {', '.join(missing)}")
        samples.append(
            ChangeVQASample(
                sample_id=str(row["sample_id"]),
                before_path=str(row["before_path"]),
                after_path=str(row["after_path"]),
                question=str(row["question"]),
                answer=str(row["answer"]).strip().lower(),
                question_type=str(row.get("question_type", "unspecified")),
            )
        )
    return samples


def normalize_answer(text: str) -> str:
    """Reduce an answer to a comparable short form.

    Only whitespace, case, and trailing punctuation are normalised. No synonym mapping is
    applied: deciding that "increased" and "more" mean the same thing is an evaluation policy
    choice, and burying it here would quietly change reported accuracy.
    """
    return " ".join(text.strip().lower().rstrip(".!?").split())


def score_predictions(
    samples: list[ChangeVQASample],
    predictions: list[ChangeVQAPrediction],
    *,
    note: str = "",
) -> ChangeVQAMetrics:
    """Compute accuracy, coverage, and per-type accuracy from real predictions.

    Samples with no matching prediction count as unanswered, so a harness cannot improve its
    score by dropping hard cases.
    """
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    metrics = ChangeVQAMetrics(samples=len(samples), note=note)

    grouped: dict[str, list[tuple[bool, bool]]] = defaultdict(list)

    for sample in samples:
        prediction = by_id.get(sample.sample_id)
        answered = bool(prediction and prediction.answered)
        correct = bool(
            answered
            and prediction is not None
            and normalize_answer(prediction.predicted) == normalize_answer(sample.answer)
        )
        metrics.answered += int(answered)
        metrics.correct += int(correct)
        grouped[sample.question_type].append((answered, correct))

    total = max(1, len(samples))
    metrics.accuracy = round(metrics.correct / total, 4)
    metrics.coverage = round(metrics.answered / total, 4)
    metrics.accuracy_on_answered = (
        round(metrics.correct / metrics.answered, 4) if metrics.answered else 0.0
    )

    for question_type, outcomes in sorted(grouped.items()):
        answered = sum(1 for entry in outcomes if entry[0])
        correct = sum(1 for entry in outcomes if entry[1])
        metrics.per_type[question_type] = {
            "samples": float(len(outcomes)),
            "answered": float(answered),
            "correct": float(correct),
            "accuracy": round(correct / max(1, len(outcomes)), 4),
            "coverage": round(answered / max(1, len(outcomes)), 4),
        }

    return metrics


# --- the adapter -------------------------------------------------------------------


def _short_answer(question, change) -> tuple[str, bool]:
    """Map a measurement onto a short answer token.

    Returns ``(answer, answered)``. The mapping is deliberately narrow: a token is only
    produced where the measurement genuinely determines it.

    ``builtup`` questions always decline. No built-up or land-use classifier exists in this
    system, so answering "yes, urban area increased" from a brightness change would be a
    fabricated classification - exactly the failure this project forbids, and it would inflate
    a benchmark score with guesses.

    Mapping these tokens onto a particular dataset's answer vocabulary is a policy decision and
    belongs in the Phase 9 harness, not here.
    """
    from ml.inference.temporal import ChangeQuestion, magnitude_band

    if question is ChangeQuestion.OCCURRED:
        return ("yes" if change.significant else "no"), True

    if question is ChangeQuestion.MAGNITUDE:
        return magnitude_band(change), True

    if question is ChangeQuestion.WHERE:
        if not change.regions:
            return "no change", True
        return change.regions[0].position, True

    if question is ChangeQuestion.VEGETATION:
        direction = change.direction_for("NDVI")
        return (direction, True) if direction else (UNANSWERABLE, False)

    if question is ChangeQuestion.WATER:
        direction = change.direction_for("NDWI")
        return (direction, True) if direction else (UNANSWERABLE, False)

    if question is ChangeQuestion.WHICH_DATE:
        for name in ("NDVI", "NDWI"):
            index = change.index_by_name(name)
            if index is not None and index.available and index.delta is not None:
                if index.delta > 0:
                    return "after", True
                if index.delta < 0:
                    return "before", True
                return "same", True
        return UNANSWERABLE, False

    # WHAT is prose, and UNSUPPORTED is by definition out of scope. Neither yields a token
    # that could be compared against a short reference answer.
    return UNANSWERABLE, False


def predict_sample(sample: ChangeVQASample, **detect_kwargs: Any) -> ChangeVQAPrediction:
    """Run the real Phase 6 pipeline on one sample and return a short-form prediction.

    Uses the same co-registration, detection, and question routing as the serving path, so a
    benchmark measures the shipped behaviour rather than a parallel implementation.

    A pair that cannot be compared, or a measurement reported inconclusive, yields
    :data:`UNANSWERABLE` with the reason attached. That is counted as unanswered, never as a
    wrong answer, because the two are different failures.
    """
    from geospatial import AlignmentError, align_pair, read_metadata, resolve_band_roles
    from ml.inference.temporal import classify_change_question, detect_change

    question, _direction = classify_change_question(sample.question)

    try:
        aligned = align_pair(sample.before_path, sample.after_path)
    except AlignmentError as exc:
        return ChangeVQAPrediction(
            sample_id=sample.sample_id,
            predicted=UNANSWERABLE,
            answered=False,
            question_kind=str(question),
            blocked_reason=f"{exc.code}: {exc.message}",
        )

    change = detect_change(
        aligned,
        roles_first=resolve_band_roles(read_metadata(sample.before_path)),
        roles_second=resolve_band_roles(read_metadata(sample.after_path)),
        **detect_kwargs,
    )

    if change.inconclusive:
        return ChangeVQAPrediction(
            sample_id=sample.sample_id,
            predicted=UNANSWERABLE,
            answered=False,
            question_kind=str(question),
            changed_fraction=0.0,
            blocked_reason=change.inconclusive_reason,
        )

    predicted, answered = _short_answer(question, change)
    return ChangeVQAPrediction(
        sample_id=sample.sample_id,
        predicted=predicted,
        answered=answered,
        question_kind=str(question),
        changed_fraction=round(change.changed_fraction, 5),
        detail=f"{change.changed_pixels} of {change.valid_pixels} valid px changed",
    )


def predict_all(samples: list[ChangeVQASample], **detect_kwargs: Any) -> list[ChangeVQAPrediction]:
    """Predict every sample. Errors are recorded per sample, not raised for the whole run."""
    predictions: list[ChangeVQAPrediction] = []
    for sample in samples:
        try:
            predictions.append(predict_sample(sample, **detect_kwargs))
        except Exception as exc:  # noqa: BLE001 - one bad raster must not void the run
            predictions.append(
                ChangeVQAPrediction(
                    sample_id=sample.sample_id,
                    predicted=UNANSWERABLE,
                    answered=False,
                    blocked_reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return predictions
