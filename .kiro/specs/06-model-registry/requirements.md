# Spec 06 — Model / Tool Registry (stub)

> Phase 3 (seeded in Phase 1). Full `design.md` + `tasks.md` when the phase begins.

## Introduction
A central registry of tools/models the agent may select from. Seeded in Phase 1 with
honest `planned` statuses; entries flip to `available` when a real implementation and
checkpoint exist.

## Requirements (high level)
- Each entry: name, version, task, modalities, input contract, output contract,
  allowed parameters, device requirements, checkpoint, license, health status.
- Logical tools: VQA, Captioning, Grounding, TemporalChange, ChangeVQA,
  OpticalSARFusion, RasterPreprocessor, RegistrationValidator, EvidenceAggregator,
  ConfidenceCalibrator.
- The agent may ONLY select from the registry; parameters validated against schemas.

## Design (outline)
`registry/models.py` (ToolSpec + registry), `registry/tasks.py` (task types). Health
check reports availability (checkpoint present, device ok). Exposed via `/models`, `/tasks`.

## Implementation tasks (outline)
Define ToolSpec + schemas → seed entries (planned) → health checks → per-tool parameter
allowlists → tests → flip statuses as specialists land.

## Dependencies
Backend core (01). Consumed by agent (05) and specialists (08–11).

## Acceptance criteria
- `/models` and `/tasks` reflect real, honest capability status.
- Unregistered tools/parameters are rejected.

## Tests
Registry schema/contract tests; parameter-allowlist validation; health-status honesty.

## User inputs required
- Model/license confirmations when specific checkpoints are chosen (later phases).

## Known risks
- Keeping statuses honest (no `available` without a working implementation).
