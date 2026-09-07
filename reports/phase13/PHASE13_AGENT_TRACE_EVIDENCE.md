# Phase 13 — Agentic Orchestration Trace Evidence

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07
**Source:** live re-run against the running app, `reports/phase13/e2e/` (`SUMMARY.json` + per-flow dumps). Reproduce with `PHASE_E2E_OUT=reports/phase13/e2e .venv\Scripts\python.exe scripts\phase12\e2e_validate.py`.

This confirms the Phase 12 orchestration behavior is intact (no regression). Every completed flow's trace exposes the full auditable pipeline; no chain-of-thought is emitted.

## Orchestration checkpoints (all present)

| # | Checkpoint | Where observed |
|---|---|---|
| 1 | Query understanding | trace "Query understanding" (query accepted, length) |
| 2 | Classification | trace "Task classification" with inferred task |
| 3 | Routing score / signals | e.g. Flow A `vqa` routing 0.43 matched "visible","what"; Flow C `optical_sar_fusion` 0.76 |
| 4 | Validation | trace "Validation" (inputs, task shape, tool registration, parameters) |
| 5 | Tool selection | trace "Tool selection" per step with tool id + version + parameters |
| 6 | Model selection | trace "Model selection"/"Model provenance" (adapter-vs-baseline note) |
| 7 | Execution | trace "Execution (<tool>)" with the produced answer/measurements |
| 8 | Evidence combination | trace "Evidence" (typed item counts); 4/11/15 items in A/B/C |
| 9 | Confidence | trace "Confidence" (level + score) with `confidence.signals` |
| 10 | Fallback / limitations | "Limitations" warnings; Flow E "Workflow failed/stopped" |

## Per-flow confirmation (Phase 13 re-run)

- **A single-image VQA** → `completed`/`vqa`, routed **`vqa_adapted`** ("selected the Phase 5 adapted model (adapter present)"), Medium; honest uncalibrated warnings.
- **B bi-temporal** → `completed`/`change_vqa`, **deterministic** `registration_validator`→`change_vqa`; learned models **not** selected; High.
- **C optical+SAR** → `completed`/`optical_sar_fusion`, feature-level cross-modal; High.
- **E1 incompatible pair** → `rejected`, `incompatible_inputs`; workflow stops **before** any specialist executes; **no fabricated answer**.
- **E2 unsupported query** → `rejected`/`unsupported`, `answer=null`, confidence **Unable to determine**; no specialist runs.

## Routing safety (re-confirmed)

- Unsupported requests do **not** route to an incorrect specialist (E2: classification fails cleanly).
- Incompatible inputs are **rejected** (E1 + compatibility endpoint `compatible=false`).
- Learned specialists remain **non-default**: `flow_d_learned.json` → `learned_tools_on_default_toolchain=[]`, `deterministic_remains_default=true`. The learned SECOND/CDVQA checkpoints load and run only via explicit direct invocation (validation), never via the agent's default toolchain.

**Verdict:** agentic orchestration `PASS` — auditable, honest, and unchanged from Phase 12.
