# Phase 14 — Production Verification: change_model integrated as an additional specialist

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07
**Verdict:** `VERIFIED_AND_INTEGRATED_AS_ADDITIONAL_SPECIALIST`

The external trained checkpoint at `change_model/model.safetensors` is executed by the **existing** SatQuery AI application through its normal workflow, reachable by the agent, with real inference surfaced through the existing API, evidence, confidence, and trace. The deterministic default and all existing models are preserved.

## Acceptance statement (now TRUE)

> Submitting a real change question to the running SatQuery API routes the existing agent to the already-trained `change_model` checkpoint, executes the real checkpoint, returns its actual generated output through the existing API, and records it in the existing evidence/trace system.

## End-to-end proof (live API, not a standalone script)

Backend started with `CHANGE_VQA_MODEL=blip`; driven exactly as the UI does. Full capture: `reports/phase14/e2e/change_model_e2e.json`; run log `reports/phase14_e2e_run.log`.

- **Query:** "Did the scene change between the two dates?" on demo T1/T2 (real uploads).
- **Task classification:** `change_vqa` (routing score 0.47).
- **Tool selection / model selection (trace):** Step 2 → `change_vqa_blip vblip-vqa-cdvqa-external`; note: *"selected the external trained BLIP Change-VQA model (CHANGE_VQA_MODEL=blip)."*
- **Execution (trace):** `Execution (change_vqa_blip)` → *The trained Change-VQA model (external BLIP) answers: "yes" (generation confidence 0.83, uncalibrated).*
- **Answer (API):** deterministic registration statement + the real BLIP generated answer.
- **Evidence:** 6 items incl. a `model` item naming `checkpoint: change_model/model.safetensors`, architecture, raw generation, and the frozen `t1` convention.
- **Confidence:** Medium (0.518), signals include specialist strength 0.83 and the honesty warnings.
- **Checks:** `routed_to_change_vqa_blip=true`, `executed_change_vqa_blip_in_trace=true`, `status_completed=true`, `answer_present=true` → **E2E_ACCEPTANCE: PASS**.

## Checkpoint integrity (weights unchanged)

| | SHA256 |
|---|---|
| Before | `E1DD3D496B0472D9D3F9D051810BE066F440BDD4178B35D990128B6590469F2F` |
| After | `E1DD3D496B0472D9D3F9D051810BE066F440BDD4178B35D990128B6590469F2F` |
| Match | **True** |

`reports/phase14/change_model_sha256_{before,after}.txt`. Weights loaded read-only; nothing retrained.

## Integration points (all in the EXISTING architecture)

| Layer | Change (additive) |
|---|---|
| Model loader | `ml/inference/change_model_infer.py` (BlipProcessor + BlipForQuestionAnswering, frozen `t1`, `model.generate`) |
| Specialist | `backend/app/agent/specialists/change_vqa_blip.py` (uniform `SpecialistResult`) |
| Registry | `MODEL_REGISTRY["change_vqa_blip"]` (task CHANGE_VQA, DEGRADED, benchmark + limitations) |
| Specialist lookup | `SPECIALISTS["change_vqa_blip"]` |
| Routing | `planner.select_change_vqa_tool()` gated by `CHANGE_VQA_MODEL` (default deterministic); `TASK_TOOLCHAIN` unchanged |
| Config | `settings.change_vqa_model` (env `CHANGE_VQA_MODEL`) |
| API | existing `/api/v1/analysis` (no contract change) |
| Evidence / confidence / trace | existing pipeline; BLIP evidence uses existing `model`/`preview` kinds |
| Frontend | existing `AnalysisResponse` rendering (answer/confidence/evidence/trace) — no code change needed |

## Preservation & smoke tests

- Real-inference smoke test: `scripts/phase14_blip_smoke.py` → answer "yes", gen-conf 0.91 (`reports/phase14_blip_smoke2.log`).
- Full regression GREEN: backend **455/455** (447 baseline + 8 new), ruff clean, frontend typecheck/lint clean, vitest **126/126**, build clean, docker compose config valid.
- Deterministic default preserved: `TASK_TOOLCHAIN[CHANGE_VQA] == ["registration_validator","change_vqa"]`; default `CHANGE_VQA_MODEL=deterministic` keeps all prior change-VQA behaviour.
- No existing model deleted/replaced; `change_model/` weights unchanged; not retrained; not made default.

## How to use it in the running app

Start the backend with `CHANGE_VQA_MODEL=blip` (force) or `CHANGE_VQA_MODEL=auto` (classification-style questions), then ask a change question in the UI. With the default (`deterministic`) the app behaves exactly as before.
