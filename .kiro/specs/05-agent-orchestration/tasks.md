# Spec 05 — Agent Orchestration · Tasks

- [x] 1. Pipeline contracts in `types.py` (intent, plan, evidence, confidence, trace, outcome)
- [x] 2. `planner.py` — `TASK_TOOLCHAIN` for every task; registration validated first on pairs
- [x] 3. `validation.py` — `validate_plan` (structure), `validate_parameters` (allowlist),
      `check_tool_runnable` (per step), `validate_inputs_for_task`
- [x] 4. Uniform `Specialist` interface + `SpecialistContext` + `SpecialistUnavailable`
- [x] 5. `confidence.py` — signal-based assessment, `Unable to determine` when unmeasurable
- [x] 6. `trace.py` — `TraceBuilder`, safe parameter summaries, no chain-of-thought
- [x] 7. `orchestrator.py` — full pipeline, four outcome states, workflow short-circuit on
      incompatible pairs
- [x] 8. API integration (`POST /analysis` + retrieval) and DB persistence
- [x] 9. Tests: routing, plan rejection, parameter validation, confidence, trace, E2E

## Deferred to later phases
- Hard timeout enforcement / cancellation (Phase 4+, once real inference runtime matters).
- Registering Phase 4–7 specialists in `specialists/SPECIALISTS` and flipping their registry
  status to `available` as each lands.
- Report generation (`/analysis/{id}/report`) — Phase 8.
