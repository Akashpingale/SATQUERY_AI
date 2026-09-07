"""Execution trace builder.

Records what the system *observably did*: which inputs it saw, how the query was
classified, which validations ran, which tool and version executed, a safe parameter
summary, and what evidence/confidence resulted.

This is an audit log, not an explanation of reasoning. Private chain-of-thought is never
recorded here - only concrete steps and their outcomes.
"""

from __future__ import annotations

from typing import Any

from .types import AnalysisPlan, ConfidenceAssessment, EvidenceItem, InputConfiguration, TraceEntry

# Substrings that mark a parameter value as sensitive. Deliberately specific: a bare
# "token" would wrongly redact legitimate model parameters such as `max_new_tokens`.
_REDACTED_MARKERS = (
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "secret",
    "password",
    "passwd",
    "credential",
)
# Exact parameter names that are always sensitive.
_REDACTED_NAMES = {"token", "key", "auth"}


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return lowered in _REDACTED_NAMES or any(m in lowered for m in _REDACTED_MARKERS)


def summarize_parameters(parameters: dict[str, Any]) -> str:
    """Render a safe, compact parameter summary."""
    if not parameters:
        return "defaults"
    parts = []
    for key in sorted(parameters):
        value = "[redacted]" if _is_sensitive(key) else parameters[key]
        parts.append(f"{key}={value}")
    return ", ".join(parts)


class TraceBuilder:
    """Accumulates trace entries in execution order."""

    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []

    def add(self, step: str, status: str, detail: str) -> None:
        self._entries.append(TraceEntry(step=step, status=status, detail=detail))

    def entries(self) -> list[TraceEntry]:
        return list(self._entries)

    # --- standard pipeline steps -------------------------------------------------

    def input_inspection(self, inputs: InputConfiguration) -> None:
        if inputs.count == 0:
            self.add("Input inspection", "failed", "No imagery supplied.")
            return
        described = ", ".join(
            f"{image.role}:{image.filename} ({image.modality}"
            f"{', georeferenced' if image.is_georeferenced else ', no georeferencing'})"
            for image in inputs.images
        )
        self.add("Input inspection", "ok", f"{inputs.count} image(s) - {described}")

    def query_understanding(self, sanitized_query: str, blocked: list[str]) -> None:
        if blocked:
            self.add(
                "Query understanding",
                "warning",
                f"{len(blocked)} instruction-like fragment(s) were ignored; the query was "
                "treated strictly as an imagery question.",
            )
        length = len(sanitized_query)
        self.add("Query understanding", "ok", f"Query accepted ({length} characters).")

    def task_classification(self, task: str, score: float, signals: list[str]) -> None:
        detail = f"Classified as {task} (routing score {score:.2f})"
        if signals:
            detail += f"; matched on: {', '.join(signals[:5])}"
        self.add("Task classification", "ok", detail)

    def validation(self, status: str, detail: str) -> None:
        self.add("Validation", status, detail)

    def tool_selection(self, plan: AnalysisPlan) -> None:
        for step in plan.steps:
            self.add(
                "Tool selection",
                "ok",
                f"Step {step.order}: {step.tool_id} v{step.tool_version} "
                f"(parameters: {summarize_parameters(step.parameters)})",
            )
        # Record why a particular model was chosen, so preferring the adapted model - or
        # falling back to the baseline - is auditable rather than invisible.
        for note in plan.selection_notes:
            self.add("Model selection", "ok", note)

    def model_provenance(self, tool_id: str) -> None:
        """Record the base model, adapter, and training data behind the selected tool."""
        from ..registry.models import MODEL_REGISTRY

        spec = MODEL_REGISTRY.get(tool_id)
        if spec is None:
            return

        parts = [f"variant={spec.variant}"]
        if spec.base_model:
            parts.append(f"base={spec.base_model}")
        if spec.adapter_path:
            parts.append(f"adapter={spec.adapter_path}")
        if spec.training_dataset:
            parts.append(f"trained on={spec.training_dataset}")
        if spec.training_method and spec.training_method != "none":
            parts.append(f"method={spec.training_method}")
        parts.append(f"benchmark={spec.benchmark_status}")
        self.add("Model provenance", "ok", "; ".join(parts))

        if spec.limitations:
            self.add(
                "Limitations",
                "warning",
                " | ".join(spec.limitations[:3]),
            )

    def execution(self, tool_id: str, status: str, detail: str) -> None:
        self.add(f"Execution ({tool_id})", status, detail)

    def evidence(self, items: list[EvidenceItem]) -> None:
        if not items:
            self.add("Evidence", "warning", "No inspectable evidence was produced.")
            return
        kinds: dict[str, int] = {}
        for item in items:
            kinds[item.kind] = kinds.get(item.kind, 0) + 1
        described = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
        self.add("Evidence", "ok", f"{len(items)} item(s): {described}")

    def confidence(self, assessment: ConfidenceAssessment) -> None:
        score = f" (score {assessment.score:.2f})" if assessment.score is not None else ""
        self.add("Confidence", "ok", f"{assessment.level}{score}")
