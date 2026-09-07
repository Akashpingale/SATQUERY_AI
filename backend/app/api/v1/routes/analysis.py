"""Analysis endpoints: run an analysis and retrieve its result, trace, and evidence.

NOTE (security): these endpoints are unauthenticated, which is acceptable for local
development and the evaluation demo but must be addressed before public deployment.
User queries are treated as untrusted data (see ``agent.understanding.sanitize_query``).
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ....agent import run_analysis
from ....agent.types import AnalysisOutcome, AnalysisStatus
from ....db.session import get_db
from ....models_db.analysis import Analysis
from ....models_db.raster_asset import RasterAsset
from ....reports import build_report_filename, render_analysis_report
from ....schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    ConfidenceOut,
    EvidenceOut,
    EvidenceResponse,
    HistoryItem,
    HistoryResponse,
    IssueOut,
    PlanOut,
    PlanStepOut,
    TraceEntryOut,
    TraceResponse,
)
from ....security.files import UnsafePathError
from ....services import storage

router = APIRouter(tags=["analysis"])

#: Evidence tokens are server-generated hex identifiers. Anything else is rejected before
#: it reaches the filesystem.
_SAFE_TOKEN = re.compile(r"[0-9a-f]{8,32}")


def _persist(db: Session, outcome: AnalysisOutcome, asset_ids: list[str]) -> Analysis:
    """Store an analysis outcome as metadata (no raster binaries in the database)."""
    record = Analysis(
        status=str(outcome.status),
        task=str(outcome.task),
        query=outcome.query,
        model_id=outcome.plan.steps[-1].tool_id if outcome.plan and outcome.plan.steps else None,
        model_version=(
            outcome.plan.steps[-1].tool_version if outcome.plan and outcome.plan.steps else None
        ),
        confidence=outcome.confidence.score,
        confidence_level=str(outcome.confidence.level),
        input_metadata={
            "asset_ids": asset_ids,
            "answer": outcome.answer,
            "confidence_signals": outcome.confidence.signals,
            "warnings": outcome.warnings,
        },
        evidence_refs={"items": [item.model_dump() for item in outcome.evidence]},
        trace={"entries": [entry.model_dump() for entry in outcome.trace]},
        error={"issues": [dict(issue) for issue in outcome.issues]} if outcome.issues else None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def _to_response(record: Analysis) -> AnalysisResponse:
    """Rebuild the API response from a persisted record."""
    input_metadata = record.input_metadata or {}
    evidence_payload = (record.evidence_refs or {}).get("items", [])
    trace_payload = (record.trace or {}).get("entries", [])
    issues_payload = (record.error or {}).get("issues", [])

    return AnalysisResponse(
        id=record.id,
        created_at=record.created_at,
        status=record.status,
        task=record.task or "unsupported",
        query=record.query or "",
        answer=input_metadata.get("answer"),
        confidence=ConfidenceOut(
            level=record.confidence_level or "Unable to determine",
            score=record.confidence,
            signals=input_metadata.get("confidence_signals", []),
        ),
        evidence=[EvidenceOut(**item) for item in evidence_payload],
        trace=[TraceEntryOut(**entry) for entry in trace_payload],
        plan=None,
        issues=[IssueOut(**issue) for issue in issues_payload],
        warnings=input_metadata.get("warnings", []),
        input_asset_ids=input_metadata.get("asset_ids", []),
    )


def _get_record_or_404(db: Session, analysis_id: str) -> Analysis:
    record = db.get(Analysis, analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return record


@router.post(
    "/analysis",
    response_model=AnalysisResponse,
    status_code=201,
    summary="Run an analysis from a natural-language query",
)
def create_analysis(
    request: AnalysisRequest,
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Run the agent pipeline over the supplied uploads.

    Always returns 201 with a result envelope, including when the outcome is
    ``rejected`` (bad/incompatible inputs) or ``unavailable`` (task understood but its
    specialist is not implemented yet). The status carries that information rather than
    an HTTP error, so the UI can show the reasoning, trace, and any partial evidence.
    """
    outcome = run_analysis(db, request.query, request.asset_ids)
    record = _persist(db, outcome, request.asset_ids)

    # Persisted rows do not carry the plan or confidence signals, so build the immediate
    # response from the live outcome for full fidelity.
    return AnalysisResponse(
        id=record.id,
        created_at=record.created_at,
        status=str(outcome.status),
        task=str(outcome.task),
        query=outcome.query,
        answer=outcome.answer,
        confidence=ConfidenceOut(
            level=str(outcome.confidence.level),
            score=outcome.confidence.score,
            signals=outcome.confidence.signals,
        ),
        evidence=[EvidenceOut(**item.model_dump()) for item in outcome.evidence],
        trace=[TraceEntryOut(**entry.model_dump()) for entry in outcome.trace],
        plan=(
            PlanOut(
                task=str(outcome.plan.task),
                steps=[
                    PlanStepOut(
                        order=step.order,
                        tool_id=step.tool_id,
                        tool_version=step.tool_version,
                        description=step.description,
                        parameters=step.parameters,
                    )
                    for step in outcome.plan.steps
                ],
                selection_notes=outcome.plan.selection_notes,
            )
            if outcome.plan
            else None
        ),
        issues=[IssueOut(**issue) for issue in outcome.issues],
        warnings=outcome.warnings,
        input_asset_ids=request.asset_ids,
    )


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Retrieve a stored analysis",
)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)) -> AnalysisResponse:
    return _to_response(_get_record_or_404(db, analysis_id))


@router.get(
    "/analysis/{analysis_id}/trace",
    response_model=TraceResponse,
    summary="Retrieve the execution trace",
)
def get_trace(analysis_id: str, db: Session = Depends(get_db)) -> TraceResponse:
    record = _get_record_or_404(db, analysis_id)
    entries = (record.trace or {}).get("entries", [])
    return TraceResponse(
        id=record.id,
        status=record.status,
        task=record.task or "unsupported",
        trace=[TraceEntryOut(**entry) for entry in entries],
    )


@router.get(
    "/analysis/{analysis_id}/evidence",
    response_model=EvidenceResponse,
    summary="Retrieve the evidence for an analysis",
)
def get_evidence(analysis_id: str, db: Session = Depends(get_db)) -> EvidenceResponse:
    record = _get_record_or_404(db, analysis_id)
    items = (record.evidence_refs or {}).get("items", [])
    return EvidenceResponse(
        id=record.id,
        status=record.status,
        task=record.task or "unsupported",
        evidence=[EvidenceOut(**item) for item in items],
    )


@router.get(
    "/evidence/{token}",
    summary="Fetch a rendered evidence image (e.g. a grounding overlay)",
    response_class=FileResponse,
)
def get_evidence_image(token: str) -> FileResponse:
    """Serve a rendered evidence image by token.

    The token is a server-generated identifier, and the resolved path is confined to the
    evidence directory, so this cannot be used to read arbitrary files.
    """
    if not _SAFE_TOKEN.fullmatch(token):
        raise HTTPException(status_code=404, detail="Evidence image not found.")
    try:
        path = storage.absolute_path(f"{storage.EVIDENCE_DIR}/overlays/{token}.png")
    except UnsafePathError as exc:
        raise HTTPException(status_code=404, detail="Evidence image not found.") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence image not found.")
    return FileResponse(path, media_type="image/png", filename=f"{token}.png")


@router.get(
    "/analysis/{analysis_id}/report",
    response_class=HTMLResponse,
    summary="Download a self-contained HTML report for an analysis",
)
def get_report(analysis_id: str, db: Session = Depends(get_db)) -> HTMLResponse:
    """Render a downloadable HTML report from the persisted analysis.

    The report is built entirely from stored data (query, task/tool selection, answer,
    confidence and its signals, evidence with images embedded, and the observable trace) and
    is served as a file attachment. Any browser can print it to PDF.
    """
    record = _get_record_or_404(db, analysis_id)
    asset_ids = (record.input_metadata or {}).get("asset_ids", [])
    assets = [asset for asset in (db.get(RasterAsset, aid) for aid in asset_ids) if asset]

    html_document = render_analysis_report(record, assets)
    filename = build_report_filename(record)
    return HTMLResponse(
        content=html_document,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/history", response_model=HistoryResponse, summary="Recent analyses")
def get_history(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    stmt = select(Analysis).order_by(Analysis.created_at.desc()).limit(limit)
    records = list(db.execute(stmt).scalars())
    return HistoryResponse(
        count=len(records),
        items=[
            HistoryItem(
                id=record.id,
                created_at=record.created_at,
                status=record.status,
                task=record.task,
                query=record.query,
                confidence_level=record.confidence_level,
            )
            for record in records
        ],
    )


# Re-exported for tests that assert on the enum values used above.
__all__ = ["router", "AnalysisStatus"]
