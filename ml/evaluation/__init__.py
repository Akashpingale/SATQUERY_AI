"""Evaluation harnesses. Metrics are always measured, never estimated."""

from .change_vqa import (
    UNANSWERABLE,
    ChangeVQAMetrics,
    ChangeVQAPrediction,
    ChangeVQASample,
    load_samples,
    normalize_answer,
    predict_all,
    predict_sample,
    score_predictions,
)
from .retrieval import (
    RetrievalMetrics,
    encode_dataset,
    evaluate_retrieval,
    random_baseline_recall,
)

__all__ = [
    "UNANSWERABLE",
    "ChangeVQAMetrics",
    "ChangeVQAPrediction",
    "ChangeVQASample",
    "RetrievalMetrics",
    "encode_dataset",
    "evaluate_retrieval",
    "load_samples",
    "normalize_answer",
    "predict_all",
    "predict_sample",
    "random_baseline_recall",
    "score_predictions",
]
