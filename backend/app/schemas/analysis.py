"""Schemas for the analysis, trace, evidence, and history endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..agent.understanding import MAX_QUERY_LENGTH


class AnalysisRequest(BaseModel):
    """A natural-language question about one or more uploaded rasters."""

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH * 2)
    asset_ids: list[str] = Field(..., min_length=1, max_length=2)

    @field_validator("asset_ids")
    @classmethod
    def _unique_assets(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("asset_ids must be unique")
        return value


class IssueOut(BaseModel):
    code: str
    message: str
    severity: str


class EvidenceOut(BaseModel):
    kind: str
    label: str
    asset_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    reference: str | None = None


class ConfidenceOut(BaseModel):
    level: str
    score: float | None = None
    signals: list[str] = Field(default_factory=list)


class TraceEntryOut(BaseModel):
    step: str
    status: str
    detail: str


class PlanStepOut(BaseModel):
    order: int
    tool_id: str
    tool_version: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class PlanOut(BaseModel):
    task: str
    steps: list[PlanStepOut] = Field(default_factory=list)
    #: Why each model was chosen (adapted preferred, or baseline fallback).
    selection_notes: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Full analysis result.

    ``answer`` is ``null`` when no genuine result could be produced (rejected inputs or a
    specialist that is not implemented yet). No placeholder text is substituted.
    """

    id: str
    created_at: datetime
    status: str  # completed | rejected | unavailable | failed
    task: str
    query: str
    answer: str | None = None
    confidence: ConfidenceOut
    evidence: list[EvidenceOut] = Field(default_factory=list)
    trace: list[TraceEntryOut] = Field(default_factory=list)
    plan: PlanOut | None = None
    issues: list[IssueOut] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    input_asset_ids: list[str] = Field(default_factory=list)


class TraceResponse(BaseModel):
    id: str
    status: str
    task: str
    trace: list[TraceEntryOut] = Field(default_factory=list)


class EvidenceResponse(BaseModel):
    id: str
    status: str
    task: str
    evidence: list[EvidenceOut] = Field(default_factory=list)


class HistoryItem(BaseModel):
    id: str
    created_at: datetime
    status: str
    task: str | None = None
    query: str | None = None
    confidence_level: str | None = None


class HistoryResponse(BaseModel):
    count: int
    items: list[HistoryItem] = Field(default_factory=list)
