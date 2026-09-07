# Phase 14 — Change-VQA Benchmark Comparison

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07
**Rule:** no retraining, no re-tuning, Test/Test2 frozen. Numbers below are the existing measured results produced by the project's CDVQA evaluation infrastructure; they are cited, not recomputed.

## Models compared (all real, all Change-VQA)

| Model | Type | Checkpoint | CDVQA Test | CDVQA Test2 | Source |
|---|---|---|---|---|---|
| **cdvqa_trained** | Closed-vocab classifier (CLIP feats + GRU + fusion head) | `models/phase9/cdvqa/cdvqa_best.pt` | **0.7114** | **0.6632** | `reports/benchmarks/cdvqa` |
| **change_vqa_blip** (external `change_model/`) | Generative BLIP VQA (single-image) | `change_model/model.safetensors` | 0.5467 (n=600) | 0.5033 (n=600) | `reports/change_model/cdvqa_evaluation.json` |
| change_vqa (deterministic) | Measurement-grounded CVA | none (algorithmic) | not a CDVQA-vocabulary model | — | default route |
| Majority-class baseline | — | — | 0.5084 | 0.4560 | `reports/benchmarks/cdvqa` |

### Protocol (unchanged)
- The BLIP pair-presentation convention was selected on the CDVQA **validation** split only (frozen `t1`); Test/Test2 were evaluated once with the frozen choice. No test-set tuning.
- `cdvqa_trained` was evaluated on the full frozen Test/Test2 splits.
- Both used the existing eval infra (`scripts/eval_change_model_cdvqa.py`, `reports/benchmarks/cdvqa`).

## Routing decision (validation-driven, honest)

- On the held-out CDVQA benchmark, **`cdvqa_trained` (0.7114/0.6632) outperforms the external BLIP `change_model` (0.5467/0.5033)**, which in turn is only modestly above the majority baseline (0.5084/0.4560).
- Therefore the external BLIP model is **NOT** promoted to default. The **deterministic** measurement-grounded `change_vqa` remains the production default (general, robust, georeferenced); `cdvqa_trained` and now `change_vqa_blip` are registered **additional** specialists.
- `change_vqa_blip` is reachable via `CHANGE_VQA_MODEL=blip` (force) or `=auto` (classification-style "changed into what" questions only). Default `deterministic` preserves all prior behaviour (455/455 tests green).

### "Combined routing where appropriate"
Per-type accuracy (Test) shows BLIP is weakest exactly where a router might have hoped it would help — e.g. `change_to_what` 0.0192, `largest_change` 0.1026 — so there is **no evidence** that routing classification questions to BLIP improves accuracy over `cdvqa_trained`. We therefore do **not** claim a combined-routing accuracy gain. BLIP is integrated as an available, honestly-labeled generative option, not as an accuracy improvement.

## Bottom line
The integration adds a genuinely different capability (a **generative** Change-VQA model) to the app, executed for real through the agent. It does **not** beat the existing trained classifier and is not presented as doing so. Deterministic stays default; `cdvqa_trained` remains the strongest learned model on the frozen benchmark.
