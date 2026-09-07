"""Plan construction, registry constraints, and parameter validation."""

from __future__ import annotations

import pytest

from app.agent.planner import TASK_TOOLCHAIN, build_plan, planned_tools
from app.agent.types import AnalysisPlan, InputConfiguration, InputImage, PlanStep, QueryIntent
from app.agent.validation import (
    MAX_TIMEOUT_SECONDS,
    PlanRejected,
    check_tool_runnable,
    resolve_tool,
    validate_parameters,
    validate_plan,
)
from app.registry.models import MODEL_REGISTRY
from app.registry.tasks import Modality, TaskType


def _image(asset_id: str, role: str, modality: Modality) -> InputImage:
    return InputImage(
        asset_id=asset_id,
        role=role,
        modality=modality,
        filename=f"{asset_id}.tif",
        is_georeferenced=True,
    )


def single_optical() -> InputConfiguration:
    return InputConfiguration(images=[_image("a", "optical", Modality.OPTICAL)])


def temporal_pair() -> InputConfiguration:
    return InputConfiguration(
        images=[_image("t1", "t1", Modality.OPTICAL), _image("t2", "t2", Modality.OPTICAL)]
    )


def optical_sar_pair() -> InputConfiguration:
    return InputConfiguration(
        images=[_image("o", "optical", Modality.OPTICAL), _image("s", "sar", Modality.SAR)]
    )


def _intent(task: TaskType) -> QueryIntent:
    return QueryIntent(task=task, score=0.9)


# --- toolchain coverage ----------------------------------------------------------


def test_every_registry_task_except_unsupported_has_a_toolchain():
    """No task may exist without a declared toolchain, or routing would dead-end."""
    for task in TaskType:
        if task in (TaskType.UNSUPPORTED,):
            continue
        assert TASK_TOOLCHAIN.get(task), f"{task} has no toolchain"


def test_all_toolchain_entries_are_registered_tools():
    """The planner may never name a tool that is not in the registry."""
    for task, tools in TASK_TOOLCHAIN.items():
        for tool_id in tools:
            assert tool_id in MODEL_REGISTRY, f"{task} -> unregistered tool '{tool_id}'"


def test_pair_tasks_validate_registration_first():
    """Compatibility must be checked before change/fusion analysis runs."""
    for task in (TaskType.TEMPORAL_CHANGE, TaskType.CHANGE_VQA, TaskType.OPTICAL_SAR_FUSION):
        assert TASK_TOOLCHAIN[task][0] == "registration_validator"


def test_planned_tools_reports_the_chain():
    assert planned_tools(TaskType.TEMPORAL_CHANGE) == [
        "registration_validator",
        "temporal_change",
    ]


# --- plan construction ------------------------------------------------------------


def test_build_plan_orders_steps_and_attaches_inputs():
    plan = build_plan(_intent(TaskType.TEMPORAL_CHANGE), temporal_pair())
    assert [step.order for step in plan.steps] == [1, 2]
    assert plan.steps[0].input_asset_ids == ["t1", "t2"]


def test_build_plan_rejects_unknown_task():
    with pytest.raises(KeyError):
        build_plan(_intent(TaskType.UNSUPPORTED), single_optical())


# --- plan validation --------------------------------------------------------------


def test_valid_metadata_plan_passes_and_resolves_version():
    plan = build_plan(_intent(TaskType.METADATA_INSPECTION), single_optical())
    validated = validate_plan(plan, single_optical())

    assert validated.steps[0].tool_id == "raster_preprocessor"
    # Version is filled from the registry, not from the proposal.
    assert validated.steps[0].tool_version == MODEL_REGISTRY["raster_preprocessor"].version


def test_unimplemented_specialist_is_not_runnable():
    """The core honesty guard: a registry entry without an implementation cannot run.

    Runnability is checked per step (not for the whole plan) so that runnable earlier
    steps still report their findings - see ``test_plan_with_unavailable_later_step``.
    """
    # Phase 7 implemented optical_sar_fusion, so the only declared-but-unimplemented tools left
    # are the Phase 4+ support placeholders that were never wired.
    unimplemented = ("evidence_aggregator", "confidence_calibrator")
    for tool_id in unimplemented:
        with pytest.raises(PlanRejected) as exc:
            check_tool_runnable(resolve_tool(tool_id))
        assert exc.value.code == "tool_unavailable"


def test_implemented_specialists_are_runnable():
    """Phase 2 support, Phase 4 single-image, Phase 6 temporal, and Phase 7 fusion."""
    for tool_id in (
        "raster_preprocessor",
        "registration_validator",
        "vqa",
        "captioning",
        "grounding",
        "temporal_change",
        "change_vqa",
        "optical_sar_fusion",
    ):
        check_tool_runnable(resolve_tool(tool_id))  # must not raise


def test_registry_status_matches_implementation_presence():
    """A tool may never advertise runnable status without a registered implementation."""
    from app.agent.specialists import is_implemented
    from app.registry.models import MODEL_REGISTRY, ToolStatus

    for tool_id, spec in MODEL_REGISTRY.items():
        runnable_status = spec.status in (ToolStatus.AVAILABLE, ToolStatus.DEGRADED)
        if runnable_status:
            assert is_implemented(tool_id), (
                f"{tool_id} claims status '{spec.status}' but has no implementation"
            )


def test_pair_plan_validates_both_steps_structurally():
    """Validation must resolve every step, not stop at the first."""
    plan = build_plan(_intent(TaskType.TEMPORAL_CHANGE), temporal_pair())
    validated = validate_plan(plan, temporal_pair())

    assert [step.tool_id for step in validated.steps] == [
        "registration_validator",
        "temporal_change",
    ]
    # Versions come from the registry, so a plan always records what would actually run.
    assert all(step.tool_version for step in validated.steps)


def test_plan_with_unavailable_later_step_still_validates_structurally():
    """Structural validation must not hide an earlier runnable step's findings.

    Fusion is the live example now that Phase 6 landed: the registration validator runs and
    reports, and only then is the missing fusion specialist surfaced.
    """
    plan = build_plan(_intent(TaskType.OPTICAL_SAR_FUSION), optical_sar_pair())
    validated = validate_plan(plan, optical_sar_pair())

    assert [step.tool_id for step in validated.steps] == [
        "registration_validator",
        "optical_sar_fusion",
    ]


def test_unregistered_tool_is_rejected():
    plan = AnalysisPlan(
        task=TaskType.METADATA_INSPECTION,
        steps=[
            PlanStep(order=1, tool_id="magic_answer_machine", tool_version="1", description="x")
        ],
    )
    with pytest.raises(PlanRejected) as exc:
        validate_plan(plan, single_optical())
    assert exc.value.code == "tool_not_registered"


def test_empty_plan_is_rejected():
    with pytest.raises(PlanRejected) as exc:
        validate_plan(AnalysisPlan(task=TaskType.METADATA_INSPECTION, steps=[]), single_optical())
    assert exc.value.code == "empty_plan"


def test_unknown_input_reference_is_rejected():
    plan = AnalysisPlan(
        task=TaskType.METADATA_INSPECTION,
        steps=[
            PlanStep(
                order=1,
                tool_id="raster_preprocessor",
                tool_version="",
                description="x",
                input_asset_ids=["not-an-input"],
            )
        ],
    )
    with pytest.raises(PlanRejected) as exc:
        validate_plan(plan, single_optical())
    assert exc.value.code == "unknown_input"


@pytest.mark.parametrize("timeout", [0, -5, MAX_TIMEOUT_SECONDS + 1])
def test_invalid_timeout_is_rejected(timeout):
    plan = build_plan(_intent(TaskType.METADATA_INSPECTION), single_optical())
    plan.timeout_seconds = timeout
    with pytest.raises(PlanRejected) as exc:
        validate_plan(plan, single_optical())
    assert exc.value.code == "timeout_invalid"


def test_too_many_images_for_single_image_task_is_rejected():
    plan = build_plan(_intent(TaskType.VQA), single_optical())
    with pytest.raises(PlanRejected) as exc:
        validate_plan(plan, temporal_pair())
    assert exc.value.code == "too_many_images"


def test_too_few_images_for_pair_task_is_rejected():
    plan = build_plan(_intent(TaskType.TEMPORAL_CHANGE), temporal_pair())
    with pytest.raises(PlanRejected) as exc:
        validate_plan(plan, single_optical())
    assert exc.value.code == "too_few_images"


def test_fusion_without_sar_is_rejected():
    plan = build_plan(_intent(TaskType.OPTICAL_SAR_FUSION), temporal_pair())
    with pytest.raises(PlanRejected) as exc:
        validate_plan(plan, temporal_pair())
    assert exc.value.code == "modality_mismatch"


# --- parameter allowlist ----------------------------------------------------------


def test_undeclared_parameter_is_rejected():
    """This is what stops the agent inventing knobs."""
    tool = resolve_tool("raster_preprocessor")
    with pytest.raises(PlanRejected) as exc:
        validate_parameters(tool, {"temperature": 0.9})
    assert exc.value.code == "parameter_not_allowed"


def test_declared_parameter_defaults_are_applied():
    tool = resolve_tool("raster_preprocessor")
    validated = validate_parameters(tool, {})
    assert validated["max_preview_side"] == 1024


def test_parameter_below_minimum_is_rejected():
    tool = resolve_tool("raster_preprocessor")
    with pytest.raises(PlanRejected) as exc:
        validate_parameters(tool, {"max_preview_side": 16})
    assert exc.value.code == "parameter_out_of_range"


def test_parameter_above_maximum_is_rejected():
    tool = resolve_tool("raster_preprocessor")
    with pytest.raises(PlanRejected) as exc:
        validate_parameters(tool, {"max_preview_side": 99999})
    assert exc.value.code == "parameter_out_of_range"


def test_parameter_type_is_enforced():
    tool = resolve_tool("raster_preprocessor")
    with pytest.raises(PlanRejected) as exc:
        validate_parameters(tool, {"max_preview_side": "not-a-number"})
    assert exc.value.code == "parameter_type_invalid"


def test_parameter_within_range_is_accepted_and_coerced():
    tool = resolve_tool("raster_preprocessor")
    validated = validate_parameters(tool, {"max_preview_side": "512"})
    assert validated["max_preview_side"] == 512
    assert isinstance(validated["max_preview_side"], int)


def test_tool_with_no_parameters_rejects_any():
    tool = resolve_tool("registration_validator")
    assert validate_parameters(tool, {}) == {}
    with pytest.raises(PlanRejected):
        validate_parameters(tool, {"anything": 1})
