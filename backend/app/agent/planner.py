"""Workflow planning.

Turns a classified intent into an ordered sequence of registered tool invocations. The
mapping from task to tools is explicit and data-driven, not a per-demo switch: the same
rules apply to any input that satisfies a task's declared shape.

Plans are always passed through ``validation.validate_plan`` before execution, so a plan
produced here is a *proposal* until validated.
"""

from __future__ import annotations

from ..registry.tasks import TaskType
from .types import AnalysisPlan, InputConfiguration, PlanStep, QueryIntent

# Tools that serve each task, in execution order. Tasks whose specialists arrive in later
# phases map to their eventual tool id; validation then rejects the plan with an honest
# "not available yet" instead of substituting a fake result.
TASK_TOOLCHAIN: dict[TaskType, list[str]] = {
    TaskType.METADATA_INSPECTION: ["raster_preprocessor"],
    TaskType.VQA: ["vqa"],
    TaskType.CAPTIONING: ["captioning"],
    TaskType.GROUNDING: ["grounding"],
    TaskType.TEMPORAL_CHANGE: ["registration_validator", "temporal_change"],
    TaskType.CHANGE_VQA: ["registration_validator", "change_vqa"],
    TaskType.OPTICAL_SAR_FUSION: ["registration_validator", "optical_sar_fusion"],
}

# Where an adapted model exists, the agent prefers it over the baseline. Preference order is
# declarative; :func:`select_model_for_tool` resolves it against what is actually loadable,
# so a missing adapter degrades to the baseline rather than failing.
#
# Grounding is deliberately absent: the Phase 5 adapter was trained and evaluated for
# image-text alignment, not localization, so claiming it improves grounding would be
# unsupported.
ADAPTED_PREFERENCE: dict[str, list[str]] = {
    "vqa": ["vqa_adapted", "vqa"],
    "captioning": ["captioning_adapted", "captioning"],
}

# Classification-style change questions ("what did it change INTO") that the deterministic
# measurement detector declines - these are where the generative BLIP model is used under
# CHANGE_VQA_MODEL="auto".
_CHANGE_CLASSIFICATION_TERMS = (
    "change to",
    "changed to",
    "change into",
    "changed into",
    "into what",
    "turn into",
    "turned into",
    "become",
    "became",
    "convert",
    "converted",
    "what is it now",
    "what did it",
    "type of change",
    "kind of change",
    "what class",
)


def _tool_available(module_name: str) -> bool:
    """Whether a learned Change-VQA inference backend can actually run."""
    try:
        import importlib

        module = importlib.import_module(f"ml.inference.{module_name}")
        return module.availability().available
    except Exception:  # noqa: BLE001 - absent ML extras / checkpoint means unavailable
        return False


#: Preferred learned Change-VQA model per setting -> (tool_id, inference module). The best
#: learned model on the frozen CDVQA benchmark is cdvqa_transfer (Phase 15).
_LEARNED_CHANGE_VQA = {
    "cdvqa_transfer": ("cdvqa_transfer", "cdvqa_transfer_infer"),
    "cdvqa": ("cdvqa_trained", "cdvqa_infer"),
    "blip": ("change_vqa_blip", "change_model_infer"),
}


def select_change_vqa_tool(query: str = "") -> tuple[str, str]:
    """Resolve the Change-VQA step to the deterministic detector or a learned model.

    The deterministic measurement-grounded ``change_vqa`` is the default. ``CHANGE_VQA_MODEL``
    can route to a learned model:

    - ``deterministic`` (default): always the deterministic detector.
    - ``cdvqa_transfer``: the Phase 15 improved classifier (best learned model).
    - ``cdvqa``: the Phase 9 trained classifier.
    - ``blip``: the external generative BLIP change_model.
    - ``auto``: the best available learned model for classification-style ("changed into
      what") questions; the deterministic detector for measurement-style questions.

    If the requested learned model is unavailable, routing degrades to the deterministic
    detector rather than failing. Returns ``(tool_id, selection_note)`` for the trace.
    """
    from ..core.config import get_settings

    mode = (get_settings().change_vqa_model or "deterministic").strip().lower()

    if mode in _LEARNED_CHANGE_VQA:
        tool_id, module = _LEARNED_CHANGE_VQA[mode]
        if _tool_available(module):
            return (
                tool_id,
                f"selected the learned Change-VQA model '{tool_id}' "
                f"(CHANGE_VQA_MODEL={mode}).",
            )
        return (
            "change_vqa",
            f"CHANGE_VQA_MODEL={mode} requested a learned model, but its checkpoint is "
            "unavailable; using the deterministic detector.",
        )

    if mode == "auto":
        q = (query or "").lower()
        is_classification = any(term in q for term in _CHANGE_CLASSIFICATION_TERMS)
        if is_classification:
            # Prefer the best available learned model for "changed into what" questions.
            for _setting, (tool_id, module) in _LEARNED_CHANGE_VQA.items():
                if _tool_available(module):
                    return (
                        tool_id,
                        f"selected the learned Change-VQA model '{tool_id}' for a "
                        "classification-style question (CHANGE_VQA_MODEL=auto).",
                    )
        return (
            "change_vqa",
            "used the deterministic detector (CHANGE_VQA_MODEL=auto).",
        )

    return "change_vqa", ""


def _adapter_present() -> bool:
    """Whether a real Phase 5 adapter exists on disk and can be described."""
    try:
        from ml.inference import describe_adapter

        return describe_adapter().present
    except Exception:  # noqa: BLE001 - absent ML extras means no adapter
        return False


def select_model_for_tool(tool_id: str) -> tuple[str, str]:
    """Resolve a base tool id to the model the agent should actually use.

    Returns ``(selected_tool_id, note)``. The note is recorded in the execution trace so
    that choosing the adapted model - or falling back to the baseline - is always visible.
    """
    candidates = ADAPTED_PREFERENCE.get(tool_id)
    if not candidates:
        return tool_id, ""

    from ..registry.models import MODEL_REGISTRY, ModelVariant

    for candidate in candidates:
        spec = MODEL_REGISTRY.get(candidate)
        if spec is None:
            continue
        if spec.variant is ModelVariant.ADAPTED:
            if _adapter_present():
                return candidate, "selected the Phase 5 adapted model (adapter present)"
            continue
        note = (
            "selected the baseline model (no Phase 5 adapter found on disk)"
            if candidates[0] != candidate
            else ""
        )
        return candidate, note

    return tool_id, ""

_STEP_DESCRIPTIONS: dict[str, str] = {
    "raster_preprocessor": "Read raster metadata and render a preview",
    "registration_validator": "Verify the image pair describes the same area",
    "vqa": "Answer the question about the image",
    "captioning": "Describe the scene",
    "grounding": "Locate the requested feature",
    "temporal_change": "Describe changes between the two dates",
    "change_vqa": "Answer the question about the change",
    "optical_sar_fusion": "Jointly analyze optical and SAR imagery",
}


def _parameters_for(tool_id: str, intent: QueryIntent) -> dict[str, object]:
    """Propose parameters for a tool.

    Only parameters the tool declares are proposed; ``validate_parameters`` enforces this
    independently, so a mistake here is caught rather than executed.
    """
    if tool_id == "grounding" and intent.target:
        return {}
    return {}


def build_plan(
    intent: QueryIntent, inputs: InputConfiguration, query: str = ""
) -> AnalysisPlan:
    """Build an (unvalidated) plan for a classified intent.

    Raises:
        KeyError: if the task has no declared toolchain (should not happen for
            registry-declared tasks).
    """
    toolchain = TASK_TOOLCHAIN.get(intent.task)
    if not toolchain:
        raise KeyError(f"No toolchain is declared for task '{intent.task}'.")

    asset_ids = [image.asset_id for image in inputs.images]

    steps = []
    notes: list[str] = []
    for index, base_tool_id in enumerate(toolchain, start=1):
        # The Change-VQA step may resolve to the external trained BLIP model when
        # CHANGE_VQA_MODEL selects it (deterministic is the default). This is done via the
        # model-selection layer so the declared TASK_TOOLCHAIN stays deterministic.
        if base_tool_id == "change_vqa":
            tool_id, note = select_change_vqa_tool(query)
        else:
            tool_id, note = select_model_for_tool(base_tool_id)
        if note:
            notes.append(f"{base_tool_id}: {note}")
        steps.append(
            PlanStep(
                order=index,
                tool_id=tool_id,
                tool_version="",  # resolved during validation from the registry
                description=_STEP_DESCRIPTIONS.get(base_tool_id, base_tool_id),
                parameters=_parameters_for(tool_id, intent),
                input_asset_ids=asset_ids,
            )
        )

    return AnalysisPlan(task=intent.task, steps=steps, selection_notes=notes)


def planned_tools(task: TaskType) -> list[str]:
    """Tools a task would use. Useful for explaining what is still missing."""
    return list(TASK_TOOLCHAIN.get(task, []))
