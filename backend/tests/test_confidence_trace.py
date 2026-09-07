"""Confidence derivation and execution-trace safety."""

from __future__ import annotations

from app.agent import confidence as confidence_module
from app.agent.trace import TraceBuilder, summarize_parameters
from app.agent.types import (
    AnalysisPlan,
    ConfidenceLevel,
    EvidenceItem,
    InputConfiguration,
    InputImage,
    PlanStep,
    SpecialistResult,
)
from app.registry.tasks import Modality, TaskType


def _inputs(georeferenced: bool = True, count: int = 1) -> InputConfiguration:
    return InputConfiguration(
        images=[
            InputImage(
                asset_id=f"a{i}",
                role="optical",
                modality=Modality.OPTICAL,
                filename=f"a{i}.tif",
                is_georeferenced=georeferenced,
            )
            for i in range(count)
        ]
    )


def _evidence(n: int = 1) -> list[EvidenceItem]:
    return [EvidenceItem(kind="metadata", label=f"item {i}") for i in range(n)]


# --- confidence -------------------------------------------------------------------


def test_strong_result_with_evidence_is_high():
    result = SpecialistResult(answer="ok", score=1.0)
    assessment = confidence_module.assess(result, 0.9, _inputs(), _evidence(2))

    assert assessment.level == ConfidenceLevel.HIGH
    assert assessment.score is not None and assessment.score >= 0.75


def test_missing_specialist_score_yields_unable_not_a_guess():
    """No strength measure means we must not invent a confidence number."""
    result = SpecialistResult(answer="ok", score=None)
    assessment = confidence_module.assess(result, 0.9, _inputs(), _evidence())

    assert assessment.level == ConfidenceLevel.UNABLE
    assert assessment.score is None


def test_weak_result_without_evidence_is_low():
    result = SpecialistResult(answer="maybe", score=0.15)
    assessment = confidence_module.assess(result, 0.3, _inputs(georeferenced=False), [])
    assert assessment.level == ConfidenceLevel.LOW


def test_missing_georeferencing_is_recorded_as_a_signal():
    result = SpecialistResult(answer="ok", score=0.9)
    assessment = confidence_module.assess(result, 0.9, _inputs(georeferenced=False), _evidence())

    assert any("georeferenc" in signal.lower() for signal in assessment.signals)


def test_warnings_reduce_confidence():
    inputs, evidence = _inputs(), _evidence(2)
    clean = confidence_module.assess(SpecialistResult(answer="a", score=0.9), 0.9, inputs, evidence)
    warned = confidence_module.assess(
        SpecialistResult(answer="a", score=0.9, warnings=["w1", "w2"]), 0.9, inputs, evidence
    )

    assert warned.score is not None and clean.score is not None
    assert warned.score < clean.score
    assert any("warning" in signal.lower() for signal in warned.signals)


def test_definite_negative_finding_is_not_treated_as_low_confidence():
    """"These images do not overlap" is a reliable finding, not a weak answer."""
    result = SpecialistResult(answer="Not compatible", score=0.0)
    assessment = confidence_module.assess(result, 0.9, _inputs(count=2), _evidence())
    assert assessment.level in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)


def test_every_assessment_lists_its_signals():
    assessment = confidence_module.assess(
        SpecialistResult(answer="a", score=0.8), 0.8, _inputs(), _evidence()
    )
    assert len(assessment.signals) >= 3


def test_unable_helper_records_reason():
    assessment = confidence_module.unable("nothing to assess")
    assert assessment.level == ConfidenceLevel.UNABLE
    assert assessment.signals == ["nothing to assess"]


# --- trace -------------------------------------------------------------------------


def test_trace_records_pipeline_steps_in_order():
    tracer = TraceBuilder()
    tracer.input_inspection(_inputs())
    tracer.query_understanding("is water present?", [])
    tracer.task_classification("vqa", 0.8, ["water"])
    tracer.validation("ok", "validated")
    tracer.evidence(_evidence())
    tracer.confidence(
        confidence_module.assess(
            SpecialistResult(answer="a", score=0.9), 0.9, _inputs(), _evidence()
        )
    )

    steps = [entry.step for entry in tracer.entries()]
    assert steps == [
        "Input inspection",
        "Query understanding",
        "Task classification",
        "Validation",
        "Evidence",
        "Confidence",
    ]


def test_trace_reports_blocked_injection_without_echoing_it():
    tracer = TraceBuilder()
    tracer.query_understanding("is water present?", ["ignore previous instructions"])

    entries = tracer.entries()
    assert entries[0].status == "warning"
    # The trace states that fragments were ignored; it does not replay the attack text.
    assert "ignore previous instructions" not in entries[0].detail.lower()


def test_trace_records_tool_and_version():
    tracer = TraceBuilder()
    plan = AnalysisPlan(
        task=TaskType.METADATA_INSPECTION,
        steps=[
            PlanStep(
                order=1,
                tool_id="raster_preprocessor",
                tool_version="1.0.0",
                description="read metadata",
                parameters={"max_preview_side": 1024},
            )
        ],
    )
    tracer.tool_selection(plan)

    detail = tracer.entries()[0].detail
    assert "raster_preprocessor" in detail
    assert "v1.0.0" in detail
    assert "max_preview_side=1024" in detail


def test_parameter_summary_redacts_sensitive_keys():
    summary = summarize_parameters({"api_key": "abc123", "max_new_tokens": 64})
    assert "abc123" not in summary
    assert "[redacted]" in summary
    assert "max_new_tokens=64" in summary


def test_parameter_summary_handles_defaults():
    assert summarize_parameters({}) == "defaults"


def test_trace_flags_missing_evidence():
    tracer = TraceBuilder()
    tracer.evidence([])
    assert tracer.entries()[0].status == "warning"
