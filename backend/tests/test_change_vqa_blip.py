"""Phase 14 tests: the external BLIP Change-VQA specialist integration.

These verify the integration is REAL and ADDITIVE without breaking the deterministic
default:
- the specialist is registered and in the registry as a Change-VQA tool;
- it is NOT on the default CHANGE_VQA toolchain (deterministic stays default);
- routing resolves to it only when CHANGE_VQA_MODEL selects it;
- availability/inference tests skip gracefully when the checkpoint is absent (it is large
  and gitignored), so CI without the weights stays green.

No weights are modified and no multi-GB download is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHANGE_MODEL_DIR = REPO / "change_model"
have_ckpt = (CHANGE_MODEL_DIR / "model.safetensors").is_file()
needs_ckpt = pytest.mark.skipif(not have_ckpt, reason="change_model checkpoint not present")


def test_change_vqa_blip_specialist_is_registered():
    from app.agent.specialists import SPECIALISTS

    assert "change_vqa_blip" in SPECIALISTS


def test_change_vqa_blip_in_registry_as_change_vqa_tool():
    from app.registry.models import MODEL_REGISTRY
    from app.registry.tasks import TaskType

    spec = MODEL_REGISTRY["change_vqa_blip"]
    assert spec.task == TaskType.CHANGE_VQA
    assert spec.checkpoint == "change_model/model.safetensors"
    # Honest provenance: an external fine-tune, benchmarked below cdvqa_trained.
    assert "0.5467" in spec.benchmark_status
    assert spec.limitations


def test_change_vqa_blip_is_not_the_default_route():
    """Deterministic change_vqa remains the default; the toolchain is unchanged."""
    from app.agent.planner import TASK_TOOLCHAIN
    from app.registry.tasks import TaskType

    assert TASK_TOOLCHAIN[TaskType.CHANGE_VQA] == ["registration_validator", "change_vqa"]
    assert "change_vqa_blip" not in TASK_TOOLCHAIN[TaskType.CHANGE_VQA]


def test_default_setting_keeps_deterministic_change_vqa():
    """With the default CHANGE_VQA_MODEL, routing resolves to the deterministic detector."""
    from app.agent.planner import select_change_vqa_tool

    tool_id, _note = select_change_vqa_tool("What changed between these two images?")
    assert tool_id == "change_vqa"


@needs_ckpt
def test_blip_mode_routes_to_change_vqa_blip(monkeypatch):
    """When CHANGE_VQA_MODEL=blip and the checkpoint is present, routing selects BLIP."""
    from app.agent import planner
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("CHANGE_VQA_MODEL", "blip")
    try:
        tool_id, note = planner.select_change_vqa_tool("Did anything change?")
        assert tool_id == "change_vqa_blip"
        assert "blip" in note.lower()
    finally:
        get_settings.cache_clear()


def test_auto_mode_uses_a_learned_model_only_for_classification_questions(monkeypatch):
    """auto routes classification-style questions to a learned model, measurement ones to
    the deterministic detector. Availability is monkeypatched so it runs without checkpoints.
    """
    from app.agent import planner
    from app.core.config import get_settings

    monkeypatch.setattr(planner, "_tool_available", lambda module: True)
    get_settings.cache_clear()
    monkeypatch.setenv("CHANGE_VQA_MODEL", "auto")
    try:
        classify_id, _ = planner.select_change_vqa_tool("What did the area change into?")
        measure_id, _ = planner.select_change_vqa_tool("How much of the scene changed?")
        # Best available learned model is preferred for classification questions.
        assert classify_id in {"cdvqa_transfer", "cdvqa_trained", "change_vqa_blip"}
        assert measure_id == "change_vqa"
    finally:
        get_settings.cache_clear()


def test_learned_mode_falls_back_when_checkpoint_absent(monkeypatch):
    """If a learned model is requested but unavailable, routing degrades to deterministic."""
    from app.agent import planner
    from app.core.config import get_settings

    monkeypatch.setattr(planner, "_tool_available", lambda module: False)
    get_settings.cache_clear()
    monkeypatch.setenv("CHANGE_VQA_MODEL", "blip")
    try:
        tool_id, note = planner.select_change_vqa_tool("Did anything change?")
        assert tool_id == "change_vqa"
        assert "unavailable" in note.lower()
    finally:
        get_settings.cache_clear()


@needs_ckpt
def test_change_model_inference_availability_reports_present():
    from ml.inference import change_model_infer

    assert change_model_infer.availability().available is True
