# Phase 14 — change_model Integration: PROGRESS / RESUME NOTE

**Saved:** 2026-09-07 (paused mid-integration; nothing committed — repo has no git history)
**Objective:** integrate the existing trained BLIP checkpoint at `change_model/` as an ADDITIONAL Change-VQA specialist reachable through the real SatQuery app. Preserve everything; deterministic stays default.

## Checkpoint integrity
- SHA256 BEFORE (baseline): `E1DD3D496B0472D9D3F9D051810BE066F440BDD4178B35D990128B6590469F2F`
  (`reports/phase14/change_model_sha256_before.txt`). Weights loaded read-only; must be re-checked identical at the end.

## Model truth (confirmed)
- `change_model/` = **BlipForQuestionAnswering** (single-image, 384px, generative), CDVQA fine-tune.
- Frozen pair convention = **`t1`** (selected on CDVQA Val only; `reports/change_model/cdvqa_evaluation.json`).
- Benchmark (existing, frozen): **CDVQA Test 0.5467 / Test2 0.5033** — BELOW `cdvqa_trained` (0.7114/0.6632). So it is ADDITIONAL, not a replacement. Do NOT claim it beats cdvqa_trained.

## DONE so far (code written, additive)
1. `ml/inference/change_model_infer.py` — real loader (`BlipProcessor`+`BlipForQuestionAnswering` from `change_model/`), `availability()`, `predict_single`/`predict_pair` (frozen `t1`), `model.generate` + uncalibrated mean-token-prob confidence, `_normalize_answer`.
2. `backend/app/agent/specialists/change_vqa_blip.py` — `ChangeVQABlipSpecialist` (tool_id `change_vqa_blip`); loads rasters via `load_asset_image` (works on GeoTIFF), truthful evidence/warnings.
3. Registered in `backend/app/agent/specialists/__init__.py` (SPECIALISTS + __all__).
4. Registry entry `change_vqa_blip` in `backend/app/registry/models.py` (task CHANGE_VQA, DEGRADED, phase 14, benchmark + limitations, checkpoint `change_model/model.safetensors`).
5. Config `change_vqa_model` in `backend/app/core/config.py` (env `CHANGE_VQA_MODEL`, default `deterministic`; values `deterministic|blip|auto`).
6. Planner `backend/app/agent/planner.py` — added `select_change_vqa_tool(query)` + `_change_model_available()` + `_CHANGE_CLASSIFICATION_TERMS`; `build_plan(intent, inputs, query="")` resolves the `change_vqa` step to `change_vqa_blip` when configured. **`TASK_TOOLCHAIN` unchanged** (tests pin it).
7. Orchestrator `backend/app/agent/orchestrator.py` — passes `clean_query` to `build_plan`.

## VERIFIED so far
- **Real inference smoke test PASSED** (`scripts/phase14_blip_smoke.py`, log `reports/phase14_blip_smoke2.log`): availability True; on demo T1/T2 → answer `"yes"`, raw `"yes"`, gen-confidence 0.9103, convention `t1`, device cpu, version `blip-vqa-cdvqa-external`. Confirms the checkpoint actually executes through the new code.

## REMAINING (next session) — to reach VERIFIED_AND_INTEGRATED_AS_ADDITIONAL_SPECIALIST
1. **Run ruff** on new/edited files; fix any lint.
2. **Full regression** (must stay green): backend pytest (baseline 447 — expect 447 + any new tests), ruff, frontend typecheck/lint/vitest 126/build, docker config. Default `change_vqa_model=deterministic` must keep all existing change_vqa tests passing (esp. `test_cdvqa.py::test_cdvqa_trained_in_registry_but_not_default_route` which pins `TASK_TOOLCHAIN[CHANGE_VQA] == ["registration_validator","change_vqa"]`, and `test_temporal_api`/`test_analysis_api` change queries).
3. **Add tests** (additive): `change_vqa_blip` registered + NOT in default toolchain; registry entry present/DEGRADED; availability; routing resolves to `change_vqa_blip` when `CHANGE_VQA_MODEL=blip`; deterministic when default.
4. **END-TO-END PROOF through the live app** (the acceptance condition): restart backend with `CHANGE_VQA_MODEL=blip` (env), then POST a real change question via `/api/v1/analysis` on demo T1/T2 → confirm the trace/plan shows `change_vqa_blip` selected, real BLIP answer returned, evidence + confidence + trace populated, through the existing API contract the UI consumes. Save the full JSON to `reports/phase14/e2e/`. (Backend currently running WITHOUT the env: `term_1788713913556_4t99jmycgky`; frontend `term_1788713913638_ihyzv8ofo2`. Need to stop+restart backend with the env for the proof.)
5. **Benchmark comparison doc** (use existing frozen numbers; do NOT re-tune/re-train): cdvqa_trained 0.7114/0.6632 vs change_model 0.5467/0.5033 → conclusion: cdvqa_trained better; BLIP added as opt-in additional specialist. Write to `reports/phase14/` (e.g. `05_BENCHMARK_RESULTS.md` / `benchmark_results.json`).
6. **SHA256 AFTER** == BEFORE (assert identical) → `change_model_sha256_after.txt`.
7. **Frontend note:** no code change needed — new specialist flows through existing AnalysisResponse (answer/confidence/evidence/trace) rendering. Verify the API response is UI-consumable (evidence kinds `model`/`preview` already supported).
8. Add `!reports/phase14/` to `.gitignore` (like phase12/13) so evidence is tracked. Keep e2e raw dumps small.
9. Write `09_PRODUCTION_VERIFICATION.md` + `production_verification.json` and update `02_MODEL_CAPABILITY_MATRIX.md` / registry-audit to include `change_vqa_blip` once E2E-proven.

## Audit deliverables already written (Phase 14 audit stage, do not redo)
`reports/phase14/`: `01_REALITY_AUDIT.md`, `reality_audit.json`, `02_MODEL_CAPABILITY_MATRIX.md`, `03_DATASET_MATRIX.md`, `GAP_ANALYSIS_AND_PLAN.md`.

## Environment facts (measured)
- GPU RTX 2050 4 GB; torch 2.6.0+cu124 CUDA available; backend runs DEVICE=cpu.
- Network: HF ~0.02 MB/s; non-HF DNS restricted → large dataset downloads infeasible (VRSBench images, RSVQA, etc. remain BLOCKED).
- Local data: SECOND 2.3 GB, LEVIR/OSCD parquet, VRSBench annotations only, CDVQA at `datasets/cdvqa/CDVQA-main` (Train/Val/Test/Test2).

## Guardrails (unchanged)
Do NOT: retrain, replace deterministic default, modify `TASK_TOOLCHAIN[CHANGE_VQA]`, alter change_model weights, remove existing models/tests, fabricate metrics. change_model integration = ADDITIONAL specialist only.
