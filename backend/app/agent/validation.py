"""Plan and parameter validation.

The agent may only use registered tools, and may only set parameters those tools declare.
Every plan passes through :func:`validate_plan` before execution; invalid plans are
rejected with an explicit reason rather than being silently repaired.
"""

from __future__ import annotations

from typing import Any

from ..registry.models import MODEL_REGISTRY, ParamSpec, ToolSpec, ToolStatus
from ..registry.tasks import TASK_REGISTRY, Modality, TaskType
from .types import AnalysisPlan, InputConfiguration, PlanStep

MAX_TIMEOUT_SECONDS = 600


class PlanRejected(ValueError):
    """Raised when a plan cannot be validated. The message is user-safe."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _coerce(value: Any, spec: ParamSpec) -> Any:
    """Coerce a value to the declared parameter type, raising on mismatch."""
    try:
        if spec.type == "int":
            if isinstance(value, bool):
                raise TypeError("bool is not an int")
            return int(value)
        if spec.type == "float":
            if isinstance(value, bool):
                raise TypeError("bool is not a float")
            return float(value)
        if spec.type == "bool":
            if not isinstance(value, bool):
                raise TypeError("expected bool")
            return value
        return str(value)
    except (TypeError, ValueError) as exc:
        raise PlanRejected(
            "parameter_type_invalid",
            f"Parameter '{spec.name}' must be of type {spec.type}.",
        ) from exc


def validate_parameters(tool: ToolSpec, parameters: dict[str, Any]) -> dict[str, Any]:
    """Validate parameters against a tool's allowlist.

    Unknown parameter names are rejected outright - this is what stops the agent from
    inventing knobs. Declared parameters are type-coerced and range/choice checked, and
    missing ones fall back to their declared defaults.
    """
    allowed = {spec.name: spec for spec in tool.allowed_parameters}

    unknown = sorted(set(parameters) - set(allowed))
    if unknown:
        permitted = ", ".join(sorted(allowed)) or "none"
        raise PlanRejected(
            "parameter_not_allowed",
            f"Tool '{tool.id}' does not accept parameter(s): {', '.join(unknown)}. "
            f"Allowed: {permitted}.",
        )

    validated: dict[str, Any] = {}
    for name, spec in allowed.items():
        if name in parameters:
            value = _coerce(parameters[name], spec)
            if spec.minimum is not None and float(value) < spec.minimum:
                raise PlanRejected(
                    "parameter_out_of_range",
                    f"Parameter '{name}' must be >= {spec.minimum}.",
                )
            if spec.maximum is not None and float(value) > spec.maximum:
                raise PlanRejected(
                    "parameter_out_of_range",
                    f"Parameter '{name}' must be <= {spec.maximum}.",
                )
            if spec.choices and str(value) not in spec.choices:
                raise PlanRejected(
                    "parameter_invalid_choice",
                    f"Parameter '{name}' must be one of: {', '.join(spec.choices)}.",
                )
            validated[name] = value
        elif spec.default is not None:
            validated[name] = spec.default
    return validated


def resolve_tool(tool_id: str) -> ToolSpec:
    """Look up a tool, rejecting anything not in the registry."""
    tool = MODEL_REGISTRY.get(tool_id)
    if tool is None:
        raise PlanRejected(
            "tool_not_registered",
            f"'{tool_id}' is not a registered tool. The agent may only use registered tools.",
        )
    return tool


def check_tool_runnable(tool: ToolSpec) -> None:
    """Raise if a registered tool cannot actually execute.

    Checked per step at execution time rather than for the whole plan up front, so that
    runnable earlier steps still produce their findings. That matters: an incompatible
    image pair should be reported as incompatible, not hidden behind a later step whose
    specialist has not been built yet.

    Raises:
        PlanRejected: with code ``tool_unavailable``.
    """
    if tool.status in (ToolStatus.PLANNED, ToolStatus.UNAVAILABLE):
        raise PlanRejected(
            "tool_unavailable",
            f"The '{tool.name}' specialist is not available yet (status: {tool.status}).",
        )


def validate_inputs_for_task(task: TaskType, inputs: InputConfiguration) -> None:
    """Check the input configuration against the task's declared shape."""
    spec = TASK_REGISTRY.get(task)
    if spec is None:
        raise PlanRejected("unknown_task", f"'{task}' is not a supported task.")

    if inputs.count < spec.min_images:
        raise PlanRejected(
            "too_few_images",
            f"{spec.title} needs at least {spec.min_images} image(s); {inputs.count} supplied.",
        )
    if inputs.count > spec.max_images:
        raise PlanRejected(
            "too_many_images",
            f"{spec.title} accepts at most {spec.max_images} image(s); {inputs.count} supplied.",
        )

    if Modality.ANY not in spec.modalities:
        # UNKNOWN is tolerated: modality detection is a hint, so an undetermined input
        # is not grounds for refusing a task the specialist could still handle.
        tolerated = set(spec.modalities) | {Modality.ANY, Modality.UNKNOWN}
        unsupported = [
            image.filename for image in inputs.images if image.modality not in tolerated
        ]
        if unsupported:
            allowed = ", ".join(str(m) for m in spec.modalities)
            raise PlanRejected(
                "modality_unsupported",
                f"{spec.title} supports {allowed} imagery; incompatible input: "
                f"{', '.join(unsupported)}.",
            )

    if task == TaskType.OPTICAL_SAR_FUSION and not inputs.has_optical_sar_pair():
        raise PlanRejected(
            "modality_mismatch",
            "Optical-SAR analysis needs one optical image and one SAR image.",
        )


def validate_plan(plan: AnalysisPlan, inputs: InputConfiguration) -> AnalysisPlan:
    """Fully validate a plan: task shape, tool registration, availability, parameters.

    Returns a normalized copy with coerced parameters and resolved tool versions.
    """
    validate_inputs_for_task(plan.task, inputs)

    if not plan.steps:
        raise PlanRejected("empty_plan", "The plan contains no executable steps.")

    if plan.timeout_seconds <= 0 or plan.timeout_seconds > MAX_TIMEOUT_SECONDS:
        raise PlanRejected(
            "timeout_invalid",
            f"Plan timeout must be between 1 and {MAX_TIMEOUT_SECONDS} seconds.",
        )

    known_assets = {image.asset_id for image in inputs.images}
    normalized: list[PlanStep] = []

    for index, step in enumerate(plan.steps, start=1):
        # Structural validation only. Runnability is checked per step at execution time
        # by ``check_tool_runnable``.
        tool = resolve_tool(step.tool_id)

        unknown_assets = sorted(set(step.input_asset_ids) - known_assets)
        if unknown_assets:
            raise PlanRejected(
                "unknown_input",
                f"Step {index} references unknown input(s): {', '.join(unknown_assets)}.",
            )

        normalized.append(
            PlanStep(
                order=index,
                tool_id=tool.id,
                tool_version=tool.version,
                description=step.description,
                parameters=validate_parameters(tool, step.parameters),
                input_asset_ids=list(step.input_asset_ids),
            )
        )

    # Carry the selection notes through: validation normalises steps but must not lose the
    # record of *why* each model was chosen, which the execution trace depends on.
    return AnalysisPlan(
        task=plan.task,
        steps=normalized,
        timeout_seconds=plan.timeout_seconds,
        selection_notes=list(plan.selection_notes),
    )
