# Phase 15 — Final Report: Real Training + Accuracy Improvement

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07

## FINAL STATUS: `TRAINED_AND_INTEGRATED_WITH_MEASURED_IMPROVEMENT`

Real training executed real gradient updates on real CDVQA data, produced a real checkpoint that reloads and runs, and the trained model **improves** the frozen CDVQA test accuracy over the existing baseline and is **used by the live SatQuery application** through its normal workflow. The improvement is small but consistent (val, Test, and Test2 all up; 7/8 question types up).

## 1. Objective
Improve SatQuery's real Change-VQA accuracy, using the already-trained external `change_model` (BLIP) as part of the strategy, without fabricating results, without touching the frozen test during selection, and integrating the best verified model into the production app.

## 2. Baseline models (measured, not retrained)
| Model | Test | Test2 | Source |
|---|---|---|---|
| cdvqa_trained (CLIP-rsicd features) | 0.7114 | 0.6632 | reports/benchmarks/cdvqa |
| change_model (BLIP, generative) | 0.5467 | 0.5033 | reports/change_model |

## 3. change_model usage (genuine, technically valid)
`change_model` is `BlipForQuestionAnswering` (single-image, generative). It is NOT treated as a temporal detector. It is used as a **frozen visual representation source**: its BLIP vision encoder (fine-tuned on CDVQA imagery) produces 768-d image features that are concatenated with the Phase 9 CLIP-rsicd 512-d features and fed to a newly trained classifier head. Its `model.safetensors` was read-only throughout (SHA256 unchanged).

## 4. Dataset inventory
CDVQA (`datasets/cdvqa/CDVQA-main`) over SECOND imagery (`D:/dataset/second`): Train 65,967 / Val 16,441 / Test 39,686 / Test2 31,036 QA over 2,968 unique image pairs; 19 answer classes. License: CDVQA research use + SECOND. No hidden/ISRO/SAC data.

## 5. Data split & leakage
Vocabularies built from **Train only**. Image-level leakage check: Train/Val/Test/Test2 pairwise overlap = **0** (asserted at train time). Train fits; Val selects; Test/Test2 evaluated once.

## 6. Training method
Controlled experiment: identical pipeline to the Phase 9 baseline (same GRU+fusion head, same optimizer/epochs/seed/selection), varying **only the frozen image feature source**. Real gradient updates on the head; frozen encoders provide features.

## 7. Training configuration
Adam, lr 1e-3, 15 epochs, batch 256, dropout 0.3, seed 42, fp32, max_len 20. clip_blip head: 1,422,099 trainable params, **3,870 optimizer steps**. `reports/phase15/training_config.json`.

## 8. Training duration / 9. Device
BLIP feature extraction 877 s; clip_blip head training 99.4 s. Device: **NVIDIA RTX 2050 (4 GB)**, torch 2.6.0+cu124.

## 10-14. Trainable params / epochs / optimizer / LR / training loss
1.42M trainable; 15 epochs; Adam; 1e-3; train_loss 1.1341 → 0.6167 (real per-epoch log in `training_log.json`).

## 15. Validation metrics
clip_blip best **val 0.7200** (epoch 15) vs baseline 0.7166; blip-only ablation 0.6881 (rejected). Selection done on Val before any test evaluation.

## 16-17. Frozen Test / Test2
| Model | Test | Test2 |
|---|---|---|
| cdvqa_trained (baseline) | 0.7114 | 0.6632 |
| **cdvqa_transfer (clip_blip)** | **0.7156** | **0.6668** |
| blip-only ablation | 0.6832 | 0.6346 |
| Δ vs baseline | **+0.42 pp** | **+0.36 pp** |

## 18. Per-question-type (Test)
7/8 types improved. Largest gains: `change_to_what` +1.91 pp, `smallest_change` +1.41 pp, `change_ratio` +0.72 pp. One slight regression: `largest_change` −0.76 pp. Full table: `experiment_comparison.json`.

## 19. Failure analysis
Dominant failure mode (shared with baseline): fine quantitative reasoning — `smallest_change` 0.34, `change_ratio` 0.40 — pooled features lack spatial arithmetic. Strong: `change_or_not` 0.84. `failure_analysis.json`.

## 20. Experiment comparison
A (baseline CLIP) < C2 (CLIP+BLIP, **best**) ; B (BLIP generative) and C (BLIP-only features) are weaker. The gain comes from the **combination**, confirming change_model contributes complementary information. `experiment_results.json`.

## 21. Accuracy improvement
**+0.42 pp Test (0.7114→0.7156), +0.36 pp Test2 (0.6632→0.6668), +0.34 pp Val.** Real and consistent; modest in magnitude (reported honestly, not oversold).

## 22-23. Checkpoint / SHA256
`models/phase15/cdvqa_clip_blip/cdvqa_transfer_best.pt` (5.69 MB), SHA256 `0e17b9ca42f213c3e110d8480596cb0a570c5a7e98c4085a19c0856ecd09d330`. Reload+inference verified in a fresh process (`scripts/phase15/reload_proof.py`).

## 24. Registry integration
`MODEL_REGISTRY["cdvqa_transfer"]` (task CHANGE_VQA, DEGRADED, phase 15, benchmark + limitations, checkpoint path).

## 25. Agent routing integration
`CHANGE_VQA_MODEL` (default `deterministic`) now supports `cdvqa_transfer` | `cdvqa` | `blip` | `auto`. `planner.select_change_vqa_tool()` resolves the Change-VQA step to the chosen learned model with availability fallback to deterministic. `TASK_TOOLCHAIN` unchanged; default preserved.

## 26. API integration
Existing `/api/v1/analysis`; new specialist flows through unchanged contract.

## 27. Frontend integration
Existing `AnalysisResponse` rendering (answer/confidence/evidence/trace) — the new model's result displays with no frontend change.

## 28. E2E proof (live app)
Backend run with `CHANGE_VQA_MODEL=cdvqa_transfer`; a real change query on demo T1/T2 → trace shows `Tool selection: cdvqa_transfer` → `Model selection: selected the learned Change-VQA model 'cdvqa_transfer'` → `Execution (cdvqa_transfer): no (improved CDVQA model, confidence 0.74)`; checkpoint SHA recorded. All checks PASS. `reports/phase15/e2e/cdvqa_transfer_e2e.json`.

## 29. Regression results
Backend pytest **460/460** (455 + 5 new); ruff clean; frontend typecheck/lint clean, vitest 126/126, build clean; docker compose config valid. (See §Regression status in machine-readable results.)

## 30. Limitations
Improvement is small (~0.4 pp); closed 19-answer vocabulary; SECOND aerial domain only; ratio/counting types remain weak; confidence uncalibrated. Deterministic stays the general default; cdvqa_transfer is the best learned option, opt-in via config.

## 31. Final status
`TRAINED_AND_INTEGRATED_WITH_MEASURED_IMPROVEMENT`.

## Preservation & integrity
cdvqa_trained, SECOND, deterministic CVA, BigEarthNet adapter, fusion, grounding, VQA/captioning, and change_model all preserved. change_model weights unchanged (SHA `E1DD3D4…469F2F` before==after). New artifacts only under `models/phase15/`. No commit made.
