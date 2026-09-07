"""Phase 9 evaluation-framework tests.

All offline and deterministic. Synthetic harnesses run the REAL production algorithms
(change detection, cross-modal fusion) on tiny fixtures — no model weights, no downloads —
so they are fast and safe for CI. External benchmark datasets are never downloaded here.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml.evaluation import calibration, change_detection, cross_modal, datasets, metrics, synthetic

# --- metrics -----------------------------------------------------------------------


def test_normalize_and_exact_match():
    assert metrics.normalize_text("  Yes. ") == "yes"
    assert metrics.exact_match("Yes!", "yes")
    assert not metrics.exact_match("no", "yes")


def test_classification_metrics_counts_abstentions_against_coverage():
    # answered+correct, answered+wrong, abstained
    m = metrics.classification_metrics(
        [(True, True), (True, False), (False, False)],
        groups=["a", "a", "b"],
    )
    assert m.samples == 3
    assert m.answered == 2
    assert m.correct == 1
    assert m.accuracy == pytest.approx(1 / 3, abs=1e-3)
    assert m.accuracy_on_answered == pytest.approx(0.5, abs=1e-3)
    assert m.coverage == pytest.approx(2 / 3, abs=1e-3)
    assert m.per_group["a"]["samples"] == 2.0


def test_segmentation_metrics_perfect_and_empty():
    ref = np.zeros((8, 8), dtype=bool)
    ref[2:5, 2:5] = True
    perfect = metrics.segmentation_metrics(ref.copy(), ref)
    assert perfect.iou == 1.0 and perfect.f1 == 1.0
    # Both empty -> perfect agreement, not a divide-by-zero.
    empty = metrics.segmentation_metrics(np.zeros((4, 4), bool), np.zeros((4, 4), bool))
    assert empty.iou == 1.0


def test_segmentation_metrics_partial_overlap():
    ref = np.zeros((4, 4), dtype=bool)
    ref[0:2, 0:2] = True  # 4 px
    pred = np.zeros((4, 4), dtype=bool)
    pred[0:2, 0:4] = True  # 8 px, 4 overlap
    seg = metrics.segmentation_metrics(pred, ref)
    assert seg.true_positive == 4
    assert seg.false_positive == 4
    assert seg.iou == pytest.approx(0.5, abs=1e-3)
    assert seg.recall == 1.0


def test_accuracy_ci_wilson_bounds():
    ci = metrics.accuracy_ci(12, 12)
    assert ci["accuracy"] == 1.0
    assert 0.0 <= ci["ci_low"] <= 1.0 <= ci["ci_high"] + 1e-9


# --- calibration -------------------------------------------------------------------


def test_reliability_perfectly_calibrated_has_low_error():
    # Confidence equals empirical accuracy in each region -> ECE ~ 0.
    conf = [0.1] * 10 + [0.9] * 10
    correct = [False] * 9 + [True] * 1 + [True] * 9 + [False] * 1
    calib = calibration.reliability(conf, correct, n_bins=10)
    assert calib.ece < 0.05
    assert calib.brier < 0.2


def test_reliability_overconfident_has_high_error():
    conf = [0.95] * 10
    correct = [False] * 10  # always wrong but very confident
    calib = calibration.reliability(conf, correct, n_bins=10)
    assert calib.ece > 0.9
    assert calib.overall_accuracy == 0.0


def test_reliability_rejects_out_of_range():
    with pytest.raises(ValueError):
        calibration.reliability([1.2], [True])


def test_selective_accuracy_monotone_when_confidence_ranks_correctness():
    conf = [0.2, 0.4, 0.6, 0.8]
    correct = [False, False, True, True]
    points = calibration.selective_accuracy(conf, correct, thresholds=[0.0, 0.5, 0.7])
    by_t = {p.threshold: p for p in points}
    assert by_t[0.0].selective_accuracy == pytest.approx(0.5, abs=1e-3)
    assert by_t[0.7].selective_accuracy == 1.0
    assert by_t[0.7].coverage < by_t[0.0].coverage


def test_abstention_quality_scores_justified_abstention():
    answered = [True, True, False, False]
    correct_if_answered = [True, False, False, True]
    q = calibration.abstention_quality(answered, correct_if_answered)
    assert q.answered_correct == 1
    assert q.answered_incorrect == 1
    assert q.abstained_would_be_incorrect == 1
    assert q.abstention_precision == pytest.approx(0.5, abs=1e-3)


# --- synthetic fixtures ------------------------------------------------------------


def test_synthetic_change_detection_fixtures_are_deterministic(tmp_path):
    a = synthetic.build_change_detection_fixtures(tmp_path / "a")
    b = synthetic.build_change_detection_fixtures(tmp_path / "b")
    assert [s.key for s in a] == [s.key for s in b]
    for sa, sb in zip(a, b, strict=True):
        assert np.array_equal(sa.changed_mask, sb.changed_mask)
    # No-change scenario has an empty mask; block scenarios do not.
    masks = {s.key: s for s in a}
    assert masks["nochange"].changed is False
    assert masks["nochange"].changed_mask.sum() == 0
    assert masks["large"].changed_mask.sum() > masks["small"].changed_mask.sum()


def test_synthetic_change_vqa_fixture_balance(tmp_path):
    samples = synthetic.build_change_vqa_fixtures(tmp_path)
    changed = [s for s in samples if s.changed]
    unchanged = [s for s in samples if not s.changed]
    assert len(changed) >= 4 and len(unchanged) >= 4
    # ids are unique
    assert len({s.sample_id for s in samples}) == len(samples)


# --- production-algorithm harnesses (real algorithms, tiny synthetic inputs) -------


def test_change_detection_harness_localizes_known_masks(tmp_path):
    result = change_detection.run(tmp_path)
    m = result["metrics"]
    # The deterministic detector should localize clear synthetic blocks well and be specific
    # on the no-change pair.
    assert result["scenarios"] == 4
    assert m["occurrence_accuracy"] == 1.0
    assert m["micro_iou"] >= 0.8
    assert m["f1"] >= 0.8


def test_cross_modal_harness_reports_measurable_properties(tmp_path):
    result = cross_modal.run(tmp_path)
    assert result["benchmark"] == "cross_modal_synthetic"
    assert result["checks_total"] == 4
    stats = result["stats"]
    # A SAR-only structured block must raise SAR structure above the flat-SAR baseline.
    assert (
        stats["sar_structure"]["sar"]["structured_fraction"]
        > stats["correlated"]["sar"]["structured_fraction"]
    )
    # Water agreement must be available with real overlap when both modalities see water.
    assert stats["water_agreement"]["water"]["available"] is True
    assert stats["water_agreement"]["water"]["agreement_iou"] > 0.0


# --- external dataset registry (availability only; never downloads) ----------------


def test_dataset_registry_lists_expected_benchmarks():
    keys = set(datasets.DATASETS)
    assert {"rsvqa", "vrsbench", "cdvqa", "second", "levir_cd", "oscd"} <= keys


def test_dataset_availability_reports_unprovisioned_without_downloading(tmp_path):
    report = datasets.availability(tmp_path)
    # Nothing is provisioned under an empty temp root; the check must be pure filesystem.
    assert report["rsvqa"]["provisioned"] is False
    assert "compatibility_note" in report["rsvqa"]
    assert report["cdvqa"]["provisioned"] is False
