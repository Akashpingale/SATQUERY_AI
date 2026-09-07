# Spec 05 — Agent Orchestration · Design (as built, Phase 3)

## Pipeline
`backend/app/agent/orchestrator.py :: run_analysis(db, query, asset_ids)`

```
1  Input inspection        load assets -> InputConfiguration (roles, modalities, CRS)
2  Query understanding     sanitize + classify (spec 04)
3  Task classification     QueryIntent, or UNSUPPORTED with a reason
4  Workflow planning       build_plan() -> ordered registered tool ids (planner.py)
5  Plan validation         task shape, tool registration, asset refs, parameters, timeout
6  Specialist execution    per step: runnability check -> specialist.run()
7  Evidence                accumulated across steps
8  Confidence              derived from real signals (confidence.py)
9  Answer synthesis        concatenated specialist answers
10 Execution trace         observable steps only (trace.py)
```

## Module layout
```
backend/app/agent/
├── types.py         pipeline contracts (intent, plan, evidence, confidence, trace, outcome)
├── understanding.py query understanding / classification (spec 04)
├── planner.py       TASK_TOOLCHAIN: task -> ordered registered tool ids
├── validation.py    PlanRejected, validate_plan, validate_parameters, check_tool_runnable
├── confidence.py    signal-based confidence assessment
├── trace.py         TraceBuilder + safe parameter summaries
├── orchestrator.py  the pipeline
└── specialists/     uniform Specialist interface + implementations
```

## Toolchains (`planner.TASK_TOOLCHAIN`)
| Task | Steps |
|---|---|
| `metadata_inspection` | `raster_preprocessor` |
| `vqa` | `vqa` |
| `captioning` | `captioning` |
| `grounding` | `grounding` |
| `temporal_change` | `registration_validator` → `temporal_change` |
| `change_vqa` | `registration_validator` → `change_vqa` |
| `optical_sar_fusion` | `registration_validator` → `optical_sar_fusion` |

Every pair task validates registration **first**, so change/fusion analysis can never run
on images that do not describe the same area. A test asserts this ordering.

## Outcome states
| Status | Meaning | `answer` | Confidence |
|---|---|---|---|
| `completed` | genuine result produced | text | derived level + score |
| `rejected` | inputs/query cannot be analyzed | reason, or `null` | `Unable to determine`, or High for a definite negative finding |
| `unavailable` | routed correctly, specialist not built yet | `null` | `Unable to determine`, score `null` |
| `failed` | unexpected execution error | `null` | `Unable to determine` |

The API returns 201 for all four; the status carries the information so the UI can show
the routing, trace, and any partial evidence rather than a bare HTTP error.

## Honesty mechanisms (the core of this phase)
1. **Registry-only tool selection.** `resolve_tool` rejects any id absent from
   `MODEL_REGISTRY`, so the agent cannot name a tool that does not exist.
2. **Allowlisted parameters.** `validate_parameters` rejects undeclared names outright and
   type/range/choice-checks the rest. This is what stops invented knobs.
3. **Per-step runnability.** `check_tool_runnable` is applied *at each step*, not to the
   whole plan. A registry entry without an implementation yields `unavailable` with the
   delivering phase named — never a stand-in answer.
4. **Derived confidence.** No specialist score means `Unable to determine`, not a guess.
5. **Trace without reasoning.** Only concrete steps, statuses, tool versions, and safe
   parameter summaries are recorded.

### Why runnability is per-step (a bug this fixed)
Validating runnability for the entire plan up front meant an incompatible temporal pair was
reported as "come back in phase 6" — because step 2 (`temporal_change`) is unbuilt — even
though step 1 (`registration_validator`) was available and would have found the real
problem. Moving the check into the execution loop means runnable earlier steps contribute
their findings first, so the user learns *"these images do not overlap"*. Verified by
`test_incompatible_temporal_pair_is_rejected_before_analysis`.

## Workflow short-circuit
When `registration_validator` reports `compatible: false`, the workflow stops and returns
`rejected` — with the comparison evidence and previews attached so the user can see why.

## Persistence
Outcomes are stored in `analyses` (metadata only): status, task, query, selected
tool/version, confidence + signals, evidence refs, trace, and issues.

## Tests
- `test_agent_planning.py` — toolchain coverage, registry constraints, plan rejection
  (unregistered tool, empty plan, unknown input, bad timeout, wrong image count, modality
  mismatch), parameter allowlist.
- `test_confidence_trace.py` — confidence derivation and trace safety.
- `test_analysis_api.py` — the full pipeline over real GeoTIFFs, all four outcome states,
  injection handling, and retrieval endpoints.

## Deferred
- Asynchronous execution/cancellation for long-running specialists (Phase 4+ when real
  inference makes runtime material). `timeout_seconds` is validated but not yet enforced
  as a hard kill.
- LLM-assisted planning; the current planner is deterministic and data-driven.
