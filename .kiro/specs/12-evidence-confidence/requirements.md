# Spec 12 — Evidence, Confidence & Trace (stub)

> Phase 4+ (cross-cutting). Stub now; elaborated as specialists land.

## Introduction
Cross-cutting layer producing machine-readable + visual evidence, a meaningful
confidence estimate, and a concise auditable execution trace for every analysis.

## Requirements (high level)
- Evidence: region/box/mask/highlight, T1/T2 comparison, optical/SAR evidence,
  metadata, specialist outputs — linked to the result.
- Confidence from real signals: model confidence, input compatibility, evidence
  agreement, modality agreement, calibration. Levels: High/Medium/Low/Unable. Never invented.
- Execution trace: task, inputs, validation, tool, model version, safe params summary,
  evidence, confidence, result. NO chain-of-thought.

## Design (outline)
`evidence/aggregator.py`, `confidence/calibrator.py`, `trace/builder.py`; consumed by
the agent (05) and surfaced via API + UI (`EvidencePanel`, `ConfidenceBadge`, `ExecutionTrace`).

## Implementation tasks (outline)
Evidence schema + aggregator → confidence signals + calibrator → trace builder →
API fields → UI components → tests.

## Dependencies
Agent (05), specialists (08–11), API/DB (13), frontend (02).

## Acceptance criteria
- Every result carries evidence refs, a justified confidence level, and a trace.
- Trace contains no private reasoning.

## Tests
Evidence linkage tests; confidence calibration/aggregation tests; trace-shape tests.

## User inputs required
- None expected.

## Known risks
- Meaningful calibration needs data; document method and limits honestly.
