# Spec 06 — Model / Tool Registry · Design (as built)

## Purpose
A single declarative source of truth for what the system can do. The agent may select
**only** from this registry, and only with parameters each tool declares. This is the
mechanism that makes invented tools and invented parameters impossible rather than merely
discouraged.

Modules: `backend/app/registry/tasks.py`, `backend/app/registry/models.py`.

## Task registry (`TASK_REGISTRY`)
`TaskSpec(type, title, description, min_images, max_images, modalities, phase)` for the
eight task types. Used as a hard constraint by both query understanding (feasibility) and
plan validation (input shape).

`Modality` = `optical | sar | any | unknown`. `unknown` exists so an undetermined input can
be represented honestly instead of being guessed into a class; no task *requires* it, and
plan validation tolerates it rather than refusing.

## Tool registry (`MODEL_REGISTRY`)
`ToolSpec(id, name, version, task, modalities, input_contract, output_contract,
allowed_parameters, device, checkpoint, license, status, phase)`.

`ToolStatus`:
| Status | Meaning |
|---|---|
| `available` | real implementation present and runnable |
| `degraded` | usable but limited (e.g. CPU baseline) |
| `planned` | declared, not yet implemented |
| `unavailable` | implemented but currently not loadable |

### Current honest state (end of Phase 3)
| Tool | Status | Version | Phase |
|---|---|---|---|
| `raster_preprocessor` | **available** | 1.0.0 | 2 |
| `registration_validator` | **available** | 1.0.0 | 2 |
| `vqa`, `captioning`, `grounding` | planned | 0.0.0 | 4 |
| `temporal_change`, `change_vqa` | planned | 0.0.0 | 6 |
| `optical_sar_fusion` | planned | 0.0.0 | 7 |
| `evidence_aggregator`, `confidence_calibrator` | planned | 0.0.0 | 4 |

`checkpoint` is `None` and `license` is `"TBD"` for anything not yet chosen — no tool
claims an artifact or licence it does not have.

## Two independent gates
A tool is runnable only when **both** hold:
1. its registry `status` is not `planned`/`unavailable` (`check_tool_runnable`), and
2. an implementation is registered in `agent/specialists/SPECIALISTS` (`is_implemented`).

Declaring a tool in the registry therefore cannot accidentally make it appear to work. A
test asserts every unimplemented tool is non-runnable and every implemented one is runnable.

## Parameter allowlist
`ParamSpec(name, type, default, minimum, maximum, choices, description)`.
`validate_parameters` rejects undeclared names, coerces declared ones to their type, and
enforces range/choice bounds. Tools with no declared parameters accept none.

## Exposure
`GET /api/v1/models` returns all entries plus a health map; `GET /api/v1/tasks` returns the
task specs. The UI and the agent read the same data.

## Tests
`test_models.py` (registry contents, honest statuses, available tools carry a real version),
`test_tasks.py` (image-count rules), `test_agent_planning.py` (toolchain entries are all
registered; runnability gates; parameter allowlist).

## Maintenance rule
When a specialist lands: add it to `SPECIALISTS`, set a real `version`, `checkpoint`, and
`license`, and flip `status` to `available`. Never flip status ahead of implementation.
