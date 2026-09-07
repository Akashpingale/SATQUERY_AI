# Phase 14 — Final Report: change_model Integration

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07

## PHASE 14 STATUS: COMPLETE (for the requested integration objective)

The already-trained external checkpoint at `change_model/model.safetensors` is now integrated as an **additional Change-VQA specialist** and is executed by the **existing** SatQuery AI application through its normal workflow. Verdict: **`VERIFIED_AND_INTEGRATED_AS_ADDITIONAL_SPECIALIST`**.

> Note: this report covers the integration task explicitly requested. The broader Phase-14 "train new VQA/captioning/segmentation/detection models" objectives remain **BLOCKED on dataset provisioning** in this environment (see `GAP_ANALYSIS_AND_PLAN.md`); no such models were fabricated.

## What was integrated (truthfully)

- The checkpoint is **BlipForQuestionAnswering** — a **single-image, generative** VQA model fine-tuned on CDVQA. It is integrated as a generative Change-VQA specialist, **not** falsely as a two-image temporal detector. For a T1/T2 pair it presents one image under the frozen `t1` convention (selected on CDVQA validation only).
- It is an **additional, non-default** specialist. The deterministic measurement-grounded `change_vqa` stays the production default; `cdvqa_trained` remains the strongest learned model on the frozen benchmark.

## End-to-end proof (live app)

Backend run with `CHANGE_VQA_MODEL=blip`; a real change query on demo T1/T2 routed through the existing agent to `change_vqa_blip`, executed the real checkpoint, and returned its generated answer with evidence/confidence/trace via the existing API. All acceptance checks PASS. Evidence: `09_PRODUCTION_VERIFICATION.md`, `reports/phase14/e2e/change_model_e2e.json`.

- Trace: `change_vqa` → Tool selection `change_vqa_blip` → Model selection "selected the external trained BLIP Change-VQA model" → `Execution (change_vqa_blip)` → answer "yes" (gen-conf 0.83).
- Checkpoint SHA256 **unchanged** (before == after == `E1DD3D4…469F2F`).

## Benchmark (no retrain, no test tuning)

| Model | CDVQA Test | Test2 |
|---|---|---|
| cdvqa_trained (project) | **0.7114** | **0.6632** |
| change_vqa_blip (external BLIP) | 0.5467 | 0.5033 |
| majority baseline | 0.5084 | 0.4560 |

Conclusion: the external BLIP model does **not** beat the project's trained classifier and is **not** presented as doing so. It adds a genuinely different (generative) capability as an opt-in option. Detail: `05_BENCHMARK_RESULTS.md`.

## Preservation & regression

- Deterministic default preserved: `TASK_TOOLCHAIN[CHANGE_VQA] == ["registration_validator","change_vqa"]`; default `CHANGE_VQA_MODEL=deterministic`.
- No existing model deleted/replaced; `change_model/` weights unchanged and not retrained; external checkpoint not made default.
- Regression GREEN: backend **455/455** (447 + 8 new), ruff clean, frontend typecheck/lint clean, vitest **126/126**, build clean, docker compose config valid.

## Real capabilities (unchanged truth) + the new addition

- New: **change_vqa_blip** — real generative BLIP Change-VQA, executed via the agent, honestly labeled (single-image, below cdvqa_trained, uncalibrated confidence).
- Preserved: deterministic CVA (default), cdvqa_trained (non-default classifier), SECOND learned, optical-SAR fusion, VQA/captioning/land-cover (retrieval), grounding (OWL-ViT), agentic orchestration.

## Accuracy limitations / failure modes (honest)

- change_vqa_blip is weakest on classification/counting question types (e.g. `change_to_what` 0.0192, `largest_change` 0.1026 on Test); its confidence is an uncalibrated generation score.
- Single-image model on a temporal task via a fixed convention is a structural limitation, disclosed in evidence/warnings.

## How to use

Start the backend with `CHANGE_VQA_MODEL=blip` (force) or `=auto` (classification-style questions) and ask a change question in the UI. Default (`deterministic`) leaves prior behaviour unchanged.

## Files (additive only)

Code: `ml/inference/change_model_infer.py`, `backend/app/agent/specialists/change_vqa_blip.py`, `backend/app/agent/specialists/__init__.py`, `backend/app/registry/models.py`, `backend/app/agent/planner.py`, `backend/app/agent/orchestrator.py`, `backend/app/core/config.py`, `backend/tests/test_change_vqa_blip.py`, `scripts/phase14/e2e_change_model.py`, `scripts/phase14_blip_smoke.py`.
Reports: `reports/phase14/{05_BENCHMARK_RESULTS.md, 09_PRODUCTION_VERIFICATION.md, production_verification.json, 11_PHASE14_FINAL_REPORT.md, e2e/change_model_e2e.json, change_model_sha256_before.txt, change_model_sha256_after.txt}` + the audit-stage docs.
`.gitignore`: added `!reports/phase14/`.

## Next highest-value improvement

`CHANGE_VQA_MODEL=auto` routing (classification questions → BLIP, measurement → deterministic) is implemented and tested; if desired it could become the default after a broader qualitative review. The remaining Phase-14 training objectives (real VQA / generative captioning / segmentation / detection) require provisioning multi-GB public datasets on a machine with normal bandwidth — still BLOCKED here.
