"""Phase 5 tests: dataset preparation, registry provenance, variant selection, and trace.

Split by dependency so the normal suite stays fast and offline:

- **Pure logic**: manifest/preparation helpers, RGB conversion, registry provenance rules,
  planner preference, trace content. Always run.
- **Adapter-dependent**: skipped when no trained adapter is on disk.
- **GPU/inference**: skipped unless weights are cached, matching the Phase 4 convention.

No test downloads a dataset or a model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from ml.datasets.bigearthnet import (
    RGB_BANDS,
    STRETCH_PERCENTILES,
    PreparationReport,
    to_rgb_uint8,
)

from app.agent.planner import ADAPTED_PREFERENCE, select_model_for_tool
from app.agent.specialists import SPECIALISTS, is_implemented
from app.registry.models import MODEL_REGISTRY, ModelVariant, ToolStatus

# =================================================================================
# Registry provenance
# =================================================================================


def test_adapted_entries_exist_for_vqa_and_captioning():
    for tool_id in ("vqa_adapted", "captioning_adapted"):
        assert tool_id in MODEL_REGISTRY, f"{tool_id} must be registered"
        spec = MODEL_REGISTRY[tool_id]
        assert spec.variant is ModelVariant.ADAPTED


def test_baseline_and_adapted_are_distinguishable():
    """A reader must always be able to tell which model produced an answer."""
    baseline = MODEL_REGISTRY["vqa"]
    adapted = MODEL_REGISTRY["vqa_adapted"]

    assert baseline.variant is ModelVariant.BASELINE
    assert adapted.variant is ModelVariant.ADAPTED
    assert baseline.id != adapted.id
    assert baseline.version != adapted.version
    assert baseline.name != adapted.name


def test_adapted_entries_declare_full_provenance():
    for tool_id in ("vqa_adapted", "captioning_adapted"):
        spec = MODEL_REGISTRY[tool_id]
        assert spec.base_model, f"{tool_id} must name its base model"
        assert spec.adapter_path, f"{tool_id} must name its adapter path"
        assert spec.training_dataset and "BigEarthNet" in spec.training_dataset
        assert spec.training_method and "LoRA" in spec.training_method
        assert spec.limitations, f"{tool_id} must declare limitations"


def test_baseline_entries_do_not_claim_adaptation():
    """The baseline must not imply this project trained it."""
    for tool_id in ("vqa", "captioning"):
        spec = MODEL_REGISTRY[tool_id]
        assert spec.adapter_path is None
        assert spec.training_method == "none (zero-shot use of an upstream checkpoint)" or (
            spec.training_method and "none" in spec.training_method
        )


def test_no_tool_claims_to_be_benchmarked_without_evidence():
    """`benchmark_status` must never be a bare claim of having been benchmarked."""
    for tool_id, spec in MODEL_REGISTRY.items():
        status = spec.benchmark_status.lower()
        if "not benchmarked" in status:
            continue
        # Anything claiming evaluation must point at where the numbers live.
        assert "report" in status or "see " in status, (
            f"{tool_id} claims '{spec.benchmark_status}' without pointing to evidence"
        )


def test_adapted_tools_have_implementations():
    for tool_id in ("vqa_adapted", "captioning_adapted"):
        assert is_implemented(tool_id)
        assert SPECIALISTS[tool_id].tool_id == tool_id


def test_adapted_specialists_use_the_adapted_variant():
    assert SPECIALISTS["vqa_adapted"].variant == "adapted"
    assert SPECIALISTS["captioning_adapted"].variant == "adapted"
    assert SPECIALISTS["vqa"].variant == "baseline"
    assert SPECIALISTS["captioning"].variant == "baseline"


def test_adapted_tools_are_not_marked_available_outright():
    """Adapted models are real but unvalidated beyond one subset, so not `available`."""
    for tool_id in ("vqa_adapted", "captioning_adapted"):
        assert MODEL_REGISTRY[tool_id].status is ToolStatus.DEGRADED


# =================================================================================
# Planner model selection
# =================================================================================


def test_preference_lists_adapted_first():
    for base_tool, candidates in ADAPTED_PREFERENCE.items():
        assert candidates[0].endswith("_adapted")
        assert candidates[-1] == base_tool


def test_grounding_has_no_adapted_preference():
    """No grounding-specific training or evaluation was done, so none is claimed."""
    assert "grounding" not in ADAPTED_PREFERENCE


def test_selection_returns_a_registered_tool_and_a_note():
    selected, note = select_model_for_tool("vqa")
    assert selected in MODEL_REGISTRY
    assert selected in {"vqa", "vqa_adapted"}
    # Whichever is chosen, the reason is recorded for the trace.
    if selected == "vqa_adapted":
        assert "adapted" in note.lower()
    else:
        assert note == "" or "baseline" in note.lower()


def test_selection_passes_through_unmapped_tools():
    assert select_model_for_tool("raster_preprocessor") == ("raster_preprocessor", "")
    assert select_model_for_tool("grounding") == ("grounding", "")


def test_selection_falls_back_when_no_adapter(monkeypatch):
    """A missing adapter must degrade to the baseline, not fail."""
    import app.agent.planner as planner

    monkeypatch.setattr(planner, "_adapter_present", lambda: False)
    selected, note = planner.select_model_for_tool("vqa")

    assert selected == "vqa"
    assert "baseline" in note.lower()


def test_selection_prefers_adapted_when_adapter_present(monkeypatch):
    import app.agent.planner as planner

    monkeypatch.setattr(planner, "_adapter_present", lambda: True)
    selected, note = planner.select_model_for_tool("vqa")

    assert selected == "vqa_adapted"
    assert "adapted" in note.lower()


# =================================================================================
# Trace provenance
# =================================================================================


def test_trace_records_model_provenance_for_adapted_tool():
    from app.agent.trace import TraceBuilder

    tracer = TraceBuilder()
    tracer.model_provenance("vqa_adapted")
    entries = tracer.entries()

    provenance = next(e for e in entries if e.step == "Model provenance")
    assert "variant=adapted" in provenance.detail
    assert "clip-rsicd-v2" in provenance.detail
    assert "BigEarthNet" in provenance.detail
    assert "LoRA" in provenance.detail

    limitations = next(e for e in entries if e.step == "Limitations")
    assert limitations.status == "warning"
    assert limitations.detail


def test_trace_records_baseline_provenance_without_adapter_claims():
    from app.agent.trace import TraceBuilder

    tracer = TraceBuilder()
    tracer.model_provenance("vqa")
    provenance = next(e for e in tracer.entries() if e.step == "Model provenance")

    assert "variant=baseline" in provenance.detail
    assert "adapter=" not in provenance.detail


def test_trace_records_selection_notes():
    from app.agent.trace import TraceBuilder
    from app.agent.types import AnalysisPlan, PlanStep
    from app.registry.tasks import TaskType

    plan = AnalysisPlan(
        task=TaskType.VQA,
        steps=[PlanStep(order=1, tool_id="vqa_adapted", tool_version="x", description="d")],
        selection_notes=["vqa: selected the Phase 5 adapted model (adapter present)"],
    )
    tracer = TraceBuilder()
    tracer.tool_selection(plan)

    selection = [e for e in tracer.entries() if e.step == "Model selection"]
    assert selection and "adapted" in selection[0].detail


# =================================================================================
# Dataset preparation helpers (no dataset files needed)
# =================================================================================


def test_rgb_conversion_stretches_to_full_range():
    """The stretch must use the real dynamic range, not a fixed reflectance ceiling.

    Measured raw values in this subset sit near 250-550, so a fixed ceiling produced
    near-black images; the stretch is verified to span the 8-bit range instead.
    """
    rng = np.random.default_rng(0)
    bands = {
        band: (rng.random((16, 16)) * 300 + 250).astype("uint16") for band in RGB_BANDS
    }

    rgb = to_rgb_uint8(bands)

    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == np.uint8
    assert rgb.max() == 255
    assert rgb.mean() > 40, "a dark result would indicate the ceiling bug regressed"


def test_rgb_conversion_requires_all_bands():
    bands = {"B04": np.ones((4, 4), dtype="uint16")}
    with pytest.raises(KeyError):
        to_rgb_uint8(bands)


def test_rgb_conversion_rejects_mismatched_shapes():
    bands = {
        "B04": np.ones((8, 8), dtype="uint16"),
        "B03": np.ones((8, 8), dtype="uint16"),
        "B02": np.ones((4, 4), dtype="uint16"),
    }
    with pytest.raises(ValueError):
        to_rgb_uint8(bands)


def test_rgb_conversion_handles_constant_image():
    """A flat patch must not divide by zero."""
    bands = {band: np.full((8, 8), 300, dtype="uint16") for band in RGB_BANDS}
    rgb = to_rgb_uint8(bands)
    assert rgb.shape == (8, 8, 3)
    assert np.all(rgb == 0)


def test_stretch_percentiles_are_sane():
    low, high = STRETCH_PERCENTILES
    assert 0 <= low < high <= 100


def test_preparation_report_records_exclusion_reasons():
    report = PreparationReport()
    report.excluded_missing_image = 3
    report.excluded_examples.append({"patch_id": "x", "reason": "image key absent"})

    assert report.excluded_missing_image == 3
    assert report.excluded_examples[0]["reason"]


# =================================================================================
# Adapter on disk (skipped when absent)
# =================================================================================


@pytest.fixture()
def adapter_present():
    from ml.inference import describe_adapter

    info = describe_adapter()
    if not info.present:
        pytest.skip(f"no Phase 5 adapter on disk: {info.reason}")
    return info


def test_adapter_reports_real_configuration(adapter_present):
    """Rank/alpha must be read from the saved config, not assumed."""
    info = adapter_present

    assert info.base_model
    assert info.rank is not None and info.rank > 0
    assert info.alpha is not None and info.alpha > 0
    assert info.target_modules, "target modules must be recorded"
    assert info.size_bytes and info.size_bytes > 1024, "adapter must contain real weights"
    assert "BigEarthNet" in (info.training_dataset or "")
    assert "LoRA" in (info.training_method or "")


def test_adapter_weight_file_exists(adapter_present):
    from pathlib import Path

    weights = list(Path(adapter_present.path).glob("adapter_model*"))
    assert weights, "adapter directory must contain a weight file"
    assert weights[0].stat().st_size > 1024


def test_variant_version_distinguishes_baseline_from_adapted(adapter_present):
    from ml.inference import variant_version

    baseline = variant_version("baseline")
    adapted = variant_version("adapted")

    assert baseline != adapted
    assert "lora" in adapted.lower()


def test_resolve_variant_selects_adapted_when_present(adapter_present):
    from ml.inference import ClipVariant, resolve_variant

    variant, note = resolve_variant(prefer_adapted=True)
    assert variant is ClipVariant.ADAPTED
    assert note

    variant, note = resolve_variant(prefer_adapted=False)
    assert variant is ClipVariant.BASELINE


# =================================================================================
# Training artefacts (skipped when a run has not happened)
# =================================================================================


@pytest.fixture()
def training_reports():
    from pathlib import Path

    reports = Path(__file__).resolve().parents[2] / "reports" / "phase5"
    if not (reports / "training_metrics.json").is_file():
        pytest.skip("no Phase 5 training metrics on disk")
    return reports


def test_training_metrics_are_real(training_reports):
    payload = json.loads((training_reports / "training_metrics.json").read_text())

    assert payload["trainable_parameters"] > 0
    assert payload["trainable_parameters"] < payload["total_parameters"]
    assert payload["epochs"], "per-epoch metrics must be recorded"
    assert payload["train_seconds"] > 0
    assert payload["peak_vram_bytes"] and payload["peak_vram_bytes"] > 0

    # Loss must be a finite positive number, and training must have actually reduced it.
    first = payload["epochs"][0]["train_loss"]
    last = payload["epochs"][-1]["train_loss"]
    assert first > 0 and last > 0
    assert last < first, "training loss should decrease across epochs"


def test_evaluation_metrics_compare_both_models(training_reports):
    path = training_reports / "evaluation_metrics.json"
    if not path.is_file():
        pytest.skip("no Phase 5 evaluation metrics on disk")
    payload = json.loads(path.read_text())

    assert payload["baseline"]["samples"] == payload["adapted"]["samples"]
    assert payload["samples"] > 0
    assert payload["split"] == "test"
    # Recall figures must be valid probabilities.
    for side in ("baseline", "adapted"):
        for key, value in payload[side].items():
            if key.startswith(("i2t_recall", "t2i_recall", "mean_recall")):
                assert 0.0 <= value <= 1.0, f"{side}.{key} = {value}"
    # Chance level must be recorded so the numbers can be read in context.
    assert payload["chance_level"]["recall_at_1"] > 0


def test_environment_record_captures_gpu_and_versions(training_reports):
    path = training_reports / "environment.json"
    if not path.is_file():
        pytest.skip("no environment record on disk")
    payload = json.loads(path.read_text())

    for key in ("python", "torch", "transformers", "peft", "seed", "batch_size_used"):
        assert payload.get(key) is not None, f"{key} must be recorded"
