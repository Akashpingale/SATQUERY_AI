"""Phase 15 tests: the improved CDVQA classifier (CLIP + change_model BLIP features).

Verifies the new trained model is registered, invocable, non-default, and reachable via
routing. Checkpoint-dependent tests skip gracefully when the Phase 15 checkpoint is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "models" / "phase15" / "cdvqa_clip_blip" / "cdvqa_transfer_best.pt"
have_ckpt = CKPT.is_file()
needs_ckpt = pytest.mark.skipif(not have_ckpt, reason="Phase 15 checkpoint not present")


def test_cdvqa_transfer_specialist_is_registered():
    from app.agent.specialists import SPECIALISTS

    assert "cdvqa_transfer" in SPECIALISTS


def test_cdvqa_transfer_in_registry_as_change_vqa_tool():
    from app.registry.models import MODEL_REGISTRY
    from app.registry.tasks import TaskType

    spec = MODEL_REGISTRY["cdvqa_transfer"]
    assert spec.task == TaskType.CHANGE_VQA
    assert spec.checkpoint == "models/phase15/cdvqa_clip_blip/cdvqa_transfer_best.pt"
    assert "0.7156" in spec.benchmark_status  # measured frozen-test accuracy


def test_cdvqa_transfer_is_not_the_default_route():
    from app.agent.planner import TASK_TOOLCHAIN
    from app.registry.tasks import TaskType

    assert TASK_TOOLCHAIN[TaskType.CHANGE_VQA] == ["registration_validator", "change_vqa"]
    assert "cdvqa_transfer" not in TASK_TOOLCHAIN[TaskType.CHANGE_VQA]


@needs_ckpt
def test_cdvqa_transfer_mode_routes_to_it(monkeypatch):
    from app.agent import planner
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("CHANGE_VQA_MODEL", "cdvqa_transfer")
    try:
        tool_id, note = planner.select_change_vqa_tool("Did anything change?")
        assert tool_id == "cdvqa_transfer"
        assert "cdvqa_transfer" in note
    finally:
        get_settings.cache_clear()


@needs_ckpt
def test_cdvqa_transfer_inference_availability_reports_present():
    from ml.inference import cdvqa_transfer_infer

    assert cdvqa_transfer_infer.availability().available is True
