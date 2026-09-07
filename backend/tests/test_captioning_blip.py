"""Tests for the external BLIP captioning specialist integration.

Verifies it is registered, in the registry as a CAPTIONING tool, the prototype default,
and that routing falls back to the CLIP captioner when the checkpoint is unavailable.
Checkpoint-dependent checks skip gracefully when caption_model/ is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "caption_model" / "model.safetensors"
have_ckpt = CKPT.is_file()
needs_ckpt = pytest.mark.skipif(not have_ckpt, reason="caption_model checkpoint not present")


def test_captioning_blip_specialist_is_registered():
    from app.agent.specialists import SPECIALISTS

    assert "captioning_blip" in SPECIALISTS


def test_captioning_blip_in_registry_as_captioning_tool():
    from app.registry.models import MODEL_REGISTRY
    from app.registry.tasks import TaskType

    spec = MODEL_REGISTRY["captioning_blip"]
    assert spec.task == TaskType.CAPTIONING
    assert spec.checkpoint == "caption_model/model.safetensors"
    assert "VRSBench" in (spec.training_dataset or "")
    # Honest: no benchmarked caption metric claimed.
    assert "not benchmarked" in spec.benchmark_status


def test_existing_clip_captioning_is_preserved():
    from app.agent.specialists import SPECIALISTS

    # The CLIP-based captioner (baseline + adapted) must remain available as alternatives.
    assert "captioning" in SPECIALISTS
    assert "captioning_adapted" in SPECIALISTS


def test_shipped_default_captioning_model_is_blip():
    """The production/prototype default (independent of the test env override) is blip."""
    from app.core.config import Settings

    assert Settings.model_fields["captioning_model"].default == "blip"


def test_default_prefers_blip_captioner_with_fallback(monkeypatch):
    from app.agent import planner
    from app.core.config import get_settings

    monkeypatch.setenv("CAPTIONING_MODEL", "blip")
    get_settings.cache_clear()
    monkeypatch.setattr(planner, "_tool_available", lambda module: True)
    try:
        tool_id, note = planner.select_captioning_tool()
        assert tool_id == "captioning_blip"
        assert "blip" in note.lower()
    finally:
        get_settings.cache_clear()

    # Fallback to the CLIP captioner when the checkpoint is unavailable.
    monkeypatch.setattr(planner, "_tool_available", lambda module: False)
    get_settings.cache_clear()
    try:
        tool_id, _ = planner.select_captioning_tool()
        assert tool_id in {"captioning", "captioning_adapted"}
    finally:
        get_settings.cache_clear()


def test_clip_mode_uses_clip_captioner(monkeypatch):
    from app.agent import planner
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("CAPTIONING_MODEL", "clip")
    try:
        tool_id, _ = planner.select_captioning_tool()
        assert tool_id in {"captioning", "captioning_adapted"}
    finally:
        get_settings.cache_clear()


@needs_ckpt
def test_caption_model_inference_availability_reports_present():
    from ml.inference import caption_model_infer

    assert caption_model_infer.availability().available is True
