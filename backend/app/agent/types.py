"""Core agent data types: input configuration, intent, plans, evidence, and trace.

These are the contracts the orchestration pipeline passes between stages. Keeping them
in one place makes the pipeline auditable and keeps the API schemas thin.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ..registry.tasks import Modality, TaskType


class ConfidenceLevel(StrEnum):
    """Confidence levels exposed to users. Never invented - always derived from signals."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNABLE = "Unable to determine"


class AnalysisStatus(StrEnum):
    COMPLETED = "completed"
    REJECTED = "rejected"  # inputs/query invalid or incompatible
    UNAVAILABLE = "unavailable"  # task understood, specialist not yet implemented
    FAILED = "failed"  # unexpected execution error


class InputImage(BaseModel):
    """One image participating in an analysis, as seen by the agent."""

    asset_id: str
    role: str  # single | optical | sar | t1 | t2
    modality: Modality
    filename: str
    width: int | None = None
    height: int | None = None
    band_count: int | None = None
    is_georeferenced: bool = False
    crs: str | None = None


class InputConfiguration(BaseModel):
    """The set of images supplied for one analysis."""

    images: list[InputImage] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.images)

    @property
    def modalities(self) -> set[Modality]:
        return {image.modality for image in self.images}

    @property
    def roles(self) -> set[str]:
        return {image.role for image in self.images}

    def by_role(self, role: str) -> InputImage | None:
        for image in self.images:
            if image.role == role:
                return image
        return None

    def has_temporal_pair(self) -> bool:
        return self.by_role("t1") is not None and self.by_role("t2") is not None

    def has_optical_sar_pair(self) -> bool:
        """True when both modalities are present in distinct images."""
        optical = [i for i in self.images if i.modality == Modality.OPTICAL]
        sar = [i for i in self.images if i.modality == Modality.SAR]
        return bool(optical) and bool(sar)


class QueryIntent(BaseModel):
    """Structured result of query understanding."""

    task: TaskType
    # 0.0-1.0 classifier score. This is a *routing* score, not an answer confidence.
    score: float
    matched_signals: list[str] = Field(default_factory=list)
    target: str | None = None  # grounding target, when the query names one
    reason: str = ""
    alternatives: list[TaskType] = Field(default_factory=list)


class PlanStep(BaseModel):
    """A single validated step in a workflow plan."""

    order: int
    tool_id: str
    tool_version: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_asset_ids: list[str] = Field(default_factory=list)


class AnalysisPlan(BaseModel):
    """A validated workflow plan. Only produced after registry + parameter validation."""

    task: TaskType
    steps: list[PlanStep] = Field(default_factory=list)
    timeout_seconds: int = 120
    #: Human-readable notes about model selection (e.g. adapted chosen, or fallback taken).
    selection_notes: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """A piece of evidence supporting a result.

    ``kind`` describes how to render/interpret it: ``metadata``, ``preview``,
    ``box``, ``mask``, ``region``, ``comparison``.
    """

    kind: str
    label: str
    asset_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    reference: str | None = None  # URL/path for image evidence


class ConfidenceAssessment(BaseModel):
    """Confidence plus the signals that produced it, so it is auditable."""

    level: ConfidenceLevel
    score: float | None = None
    signals: list[str] = Field(default_factory=list)


class TraceEntry(BaseModel):
    """One observable pipeline step. Contains no private reasoning."""

    step: str
    status: str  # ok | warning | failed | skipped
    detail: str


class SpecialistResult(BaseModel):
    """Uniform output contract every specialist must return."""

    answer: str
    score: float | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    #: Extra observable steps the specialist performed, appended to the execution trace by
    #: the orchestrator. Needed because some obligations are internal to a step - notably
    #: that resampling a temporal pair must be recorded - and the trace would otherwise only
    #: carry the step's answer. These are factual operations, never reasoning.
    trace_notes: list[TraceEntry] = Field(default_factory=list)


class AnalysisOutcome(BaseModel):
    """The complete result of one analysis run."""

    status: AnalysisStatus
    task: TaskType
    query: str
    answer: str | None = None
    confidence: ConfidenceAssessment
    evidence: list[EvidenceItem] = Field(default_factory=list)
    trace: list[TraceEntry] = Field(default_factory=list)
    plan: AnalysisPlan | None = None
    issues: list[dict[str, str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
