# Spec 05 — Agent Orchestration (stub)

> Phase 3. Stub now; full `design.md` + `tasks.md` authored when the phase begins.

## Introduction
The real orchestration layer: from a validated intent, plan and execute a workflow
over registered specialists, aggregate evidence, estimate confidence, synthesize the
answer, and emit an auditable execution trace.

## Requirements (high level)
- Pipeline: input inspection → query understanding → task classification →
  compatibility validation → model/tool selection → workflow planning → execution →
  evidence → confidence → answer synthesis → trace.
- Select ONLY registered tools with allowlisted, validated parameters; reject invalid plans.
- No hard-coded demo switch; routing generalizes to unseen inputs.
- Never expose chain-of-thought; expose a concise observable trace.

## Design (outline)
`agent/orchestrator.py` (plan+run), `agent/planner.py`, `agent/executor.py`, integrates
registry (06), specialists (08–11), evidence/confidence (12), trace builder.

## Implementation tasks (outline)
Plan schema → validator → selector → executor → evidence/confidence hooks → trace →
API `POST /analysis` integration → routing tests.

## Dependencies
Query understanding (04), registry (06), specialists (08–11), evidence/confidence (12).

## Acceptance criteria
- Correct tool/model chosen per task/modality; invalid plans rejected with clear errors.
- Trace faithfully records steps, params summary, evidence, and confidence.

## Tests
Routing-accuracy unit tests across task/modality combos; negative: invalid plan,
missing model, incompatible inputs, timeout.

## User inputs required
- Decision later: rule-based planner vs LLM-assisted planner (feasibility on CPU).

## Known risks
- Ensuring genuine (non-hardcoded) routing while keeping it reliable for demos.
