"""The orchestration pipeline.

Runs the full sequence for one analysis:

    input inspection -> query understanding -> task classification -> compatibility
    validation -> tool selection -> workflow planning -> specialist execution ->
    evidence -> confidence -> answer synthesis -> execution trace

Honesty rules enforced here:
- A task whose specialist is not implemented yet returns ``UNAVAILABLE`` with an
  explanation. It never returns an invented answer.
- Confidence is derived from real signals or reported as "Unable to determine".
- The trace records observable steps only, never private reasoning.
"""

from __future__ import annotations

from ml.inference import ModelUnavailableError
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..models_db.raster_asset import RasterAsset
from ..registry.models import MODEL_REGISTRY
from ..registry.tasks import TASK_REGISTRY, Modality, TaskType
from . import confidence as confidence_module
from .planner import build_plan, planned_tools
from .specialists import SpecialistUnavailable, get_specialist, is_implemented
from .specialists.base import SpecialistContext
from .trace import TraceBuilder
from .types import (
    AnalysisOutcome,
    AnalysisStatus,
    EvidenceItem,
    InputConfiguration,
    InputImage,
    SpecialistResult,
)
from .understanding import understand
from .validation import PlanRejected, check_tool_runnable, validate_plan

logger = get_logger("satquery.agent")


def build_input_configuration(
    db: Session, asset_ids: list[str]
) -> tuple[InputConfiguration, list[str]]:
    """Load assets into an :class:`InputConfiguration`.

    Returns the configuration plus a list of ids that could not be found.
    """
    images: list[InputImage] = []
    missing: list[str] = []

    for asset_id in asset_ids:
        asset = db.get(RasterAsset, asset_id)
        if asset is None:
            missing.append(asset_id)
            continue
        try:
            modality = Modality(asset.modality)
        except ValueError:
            modality = Modality.UNKNOWN
        images.append(
            InputImage(
                asset_id=asset.id,
                role=asset.role,
                modality=modality,
                filename=asset.original_filename,
                width=asset.width,
                height=asset.height,
                band_count=asset.band_count,
                is_georeferenced=asset.is_georeferenced,
                crs=asset.crs,
            )
        )

    return InputConfiguration(images=images), missing


def _rejected(
    tracer: TraceBuilder,
    task: TaskType,
    query: str,
    code: str,
    message: str,
) -> AnalysisOutcome:
    return AnalysisOutcome(
        status=AnalysisStatus.REJECTED,
        task=task,
        query=query,
        answer=None,
        confidence=confidence_module.unable(message),
        evidence=[],
        trace=tracer.entries(),
        issues=[{"code": code, "message": message, "severity": "error"}],
    )


def _unavailable_message(task: TaskType) -> str:
    """Explain that a task routed correctly but its specialist does not exist yet."""
    spec = TASK_REGISTRY.get(task)
    pending = [
        MODEL_REGISTRY[tool_id]
        for tool_id in planned_tools(task)
        if tool_id in MODEL_REGISTRY and not is_implemented(tool_id)
    ]
    phases = sorted({tool.phase for tool in pending})
    title = spec.title if spec else str(task)
    message = (
        f"{title} is understood and correctly routed, but its specialist is not "
        "implemented yet"
    )
    if phases:
        message += f" (arrives in phase {', '.join(str(p) for p in phases)})"
    return message + "."


def run_analysis(db: Session, raw_query: str, asset_ids: list[str]) -> AnalysisOutcome:
    """Execute one analysis end to end."""
    tracer = TraceBuilder()

    # --- 1. Input inspection ----------------------------------------------------
    inputs, missing = build_input_configuration(db, asset_ids)
    tracer.input_inspection(inputs)

    if missing:
        message = f"Unknown upload(s): {', '.join(missing)}."
        tracer.validation("failed", message)
        return _rejected(tracer, TaskType.UNSUPPORTED, raw_query, "unknown_input", message)

    if inputs.count == 0:
        message = "Upload at least one image before asking a question."
        tracer.validation("failed", message)
        return _rejected(tracer, TaskType.UNSUPPORTED, raw_query, "no_inputs", message)

    # --- 2. Query understanding + 3. task classification ------------------------
    intent, clean_query, blocked = understand(raw_query, inputs)
    tracer.query_understanding(clean_query, blocked)

    if blocked:
        logger.warning(
            "Blocked %d instruction-like fragment(s) in a query; treated as data only.",
            len(blocked),
        )

    if intent.task == TaskType.UNSUPPORTED:
        tracer.add("Task classification", "failed", intent.reason)
        return AnalysisOutcome(
            status=AnalysisStatus.REJECTED,
            task=TaskType.UNSUPPORTED,
            query=clean_query,
            answer=None,
            confidence=confidence_module.unable(intent.reason),
            trace=tracer.entries(),
            issues=[{"code": "unsupported_query", "message": intent.reason, "severity": "error"}],
        )

    tracer.task_classification(str(intent.task), intent.score, intent.matched_signals)

    # --- 4. Planning + 5. validation (task shape, tools, parameters) ------------
    try:
        proposed = build_plan(intent, inputs, clean_query)
    except KeyError as exc:
        message = str(exc)
        tracer.validation("failed", message)
        return _rejected(tracer, intent.task, clean_query, "no_toolchain", message)

    try:
        plan = validate_plan(proposed, inputs)
    except PlanRejected as exc:
        tracer.validation("failed", exc.message)
        return _rejected(tracer, intent.task, clean_query, exc.code, exc.message)

    tracer.validation("ok", "Inputs, task shape, tool registration, and parameters validated.")
    tracer.tool_selection(plan)
    for step in plan.steps:
        tracer.model_provenance(step.tool_id)

    # --- 6. Specialist execution ------------------------------------------------
    evidence: list[EvidenceItem] = []
    warnings: list[str] = []
    answers: list[str] = []
    final_result: SpecialistResult | None = None

    for step in plan.steps:
        # Runnability is checked here, per step, so earlier runnable steps have already
        # contributed their findings and evidence before we report a gap.
        tool = MODEL_REGISTRY[step.tool_id]
        try:
            check_tool_runnable(tool)
        except PlanRejected as exc:
            tracer.execution(step.tool_id, "skipped", exc.message)
            detail = _unavailable_message(intent.task)
            return AnalysisOutcome(
                status=AnalysisStatus.UNAVAILABLE,
                task=intent.task,
                query=clean_query,
                answer=None,
                confidence=confidence_module.unable(
                    "No result was produced because the specialist is unavailable."
                ),
                evidence=evidence,
                trace=tracer.entries(),
                plan=plan,
                issues=[{"code": "tool_unavailable", "message": detail, "severity": "error"}],
                warnings=warnings,
            )

        context = SpecialistContext(
            query=clean_query,
            inputs=inputs,
            parameters=step.parameters,
            db=db,
            target=intent.target,
        )
        try:
            specialist = get_specialist(step.tool_id)
            result = specialist.run(context)
        except ModelUnavailableError as exc:
            # The capability is implemented but its weights/dependencies are missing.
            # That is a deployment gap, reported as unavailable rather than as a result.
            message = (
                f"The '{step.tool_id}' model could not be loaded: {exc}. "
                "No answer is produced when the model is unavailable."
            )
            tracer.execution(step.tool_id, "failed", message)
            return AnalysisOutcome(
                status=AnalysisStatus.UNAVAILABLE,
                task=intent.task,
                query=clean_query,
                answer=None,
                confidence=confidence_module.unable(message),
                evidence=evidence,
                trace=tracer.entries(),
                plan=plan,
                issues=[{"code": "model_unavailable", "message": message, "severity": "error"}],
                warnings=warnings,
            )
        except SpecialistUnavailable as exc:
            tracer.execution(step.tool_id, "skipped", str(exc))
            return AnalysisOutcome(
                status=AnalysisStatus.UNAVAILABLE,
                task=intent.task,
                query=clean_query,
                answer=None,
                confidence=confidence_module.unable(str(exc)),
                evidence=evidence,
                trace=tracer.entries(),
                plan=plan,
                issues=[
                    {
                        "code": "tool_unavailable",
                        "message": _unavailable_message(intent.task),
                        "severity": "error",
                    }
                ],
                warnings=warnings,
            )
        except Exception:  # noqa: BLE001 - one failing step must not crash the API
            logger.exception("Specialist '%s' failed", step.tool_id)
            message = f"The '{step.tool_id}' step failed while processing the imagery."
            tracer.execution(step.tool_id, "failed", message)
            return AnalysisOutcome(
                status=AnalysisStatus.FAILED,
                task=intent.task,
                query=clean_query,
                answer=None,
                confidence=confidence_module.unable(message),
                evidence=evidence,
                trace=tracer.entries(),
                plan=plan,
                issues=[{"code": "execution_failed", "message": message, "severity": "error"}],
            )

        # Observable operations the step performed internally (e.g. reprojection and
        # resampling of a temporal pair) are recorded before its outcome, so the trace reads
        # in execution order.
        for note in result.trace_notes:
            tracer.add(note.step, note.status, note.detail)

        tracer.execution(
            step.tool_id,
            "warning" if result.warnings else "ok",
            result.answer[:200] + ("…" if len(result.answer) > 200 else ""),
        )
        evidence.extend(result.evidence)
        warnings.extend(result.warnings)
        answers.append(result.answer)
        final_result = result

        # A validator that reports incompatibility stops the workflow: running change or
        # fusion analysis on an incompatible pair would produce a meaningless result.
        if step.tool_id == "registration_validator" and result.details.get("compatible") is False:
            tracer.add(
                "Workflow",
                "failed",
                "Stopped after compatibility check: the images cannot be analyzed together.",
            )
            assessment = confidence_module.assess(result, intent.score, inputs, evidence)
            tracer.evidence(evidence)
            tracer.confidence(assessment)
            return AnalysisOutcome(
                status=AnalysisStatus.REJECTED,
                task=intent.task,
                query=clean_query,
                answer=result.answer,
                confidence=assessment,
                evidence=evidence,
                trace=tracer.entries(),
                plan=plan,
                issues=[
                    {
                        "code": "incompatible_inputs",
                        "message": result.answer,
                        "severity": "error",
                    }
                ],
                warnings=warnings,
            )

    assert final_result is not None  # a validated plan always has >= 1 step

    # --- 7. Evidence, 8. confidence, 9. answer synthesis ------------------------
    tracer.evidence(evidence)
    assessment = confidence_module.assess(final_result, intent.score, inputs, evidence)
    tracer.confidence(assessment)

    return AnalysisOutcome(
        status=AnalysisStatus.COMPLETED,
        task=intent.task,
        query=clean_query,
        answer=" ".join(answers).strip(),
        confidence=assessment,
        evidence=evidence,
        trace=tracer.entries(),
        plan=plan,
        warnings=warnings,
    )
