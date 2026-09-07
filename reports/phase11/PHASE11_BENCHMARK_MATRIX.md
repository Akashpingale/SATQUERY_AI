# Phase 11 — Benchmark Matrix

Every row is one of: **FULL TEST EVALUATION**, VALIDATION EVALUATION, SUBSET EVALUATION,
COMPATIBILITY CHECK, ❌ INCOMPATIBLE, ❌ BLOCKED, ⏭️ NOT APPLICABLE. No fabricated numbers; every
metric below comes from an actual reproducible run under `scripts/phase11/`.

| Benchmark | Task | Status | Result (real) | Notes |
|---|---|---|---|---|
| **LEVIR-CD** | binary change detection | ✅ FULL TEST EVALUATION | learned SECOND: IoU **0.0773**, F1 **0.1435**, P 0.083, R 0.527 · Otsu ref: IoU 0.0466, F1 0.0891 (n=2048) | Cross-dataset: model trained only on SECOND. Heavy over-prediction (building-only labels). Learned > deterministic here. |
| **OSCD** | binary change detection | ✅ FULL TEST EVALUATION | learned SECOND: IoU **0.1053**, F1 **0.1905** · Otsu ref: IoU **0.1453**, F1 **0.2538**, R 0.549 (n=50) | Sentinel-2 RGB crops. Deterministic **beats** the learned model → validates keeping deterministic as robust default. |
| **RSVQA** | single-image VQA | ❌ INCOMPATIBLE | — | Production VQA is zero-shot CLIP concept-scoring over a fixed vocabulary; cannot emit RSVQA yes/no/count/area answers. A constrained/generative VQA head is required. |
| **VRSBench — VQA** | single-image VQA | ❌ INCOMPATIBLE | — | Same answer-format incompatibility as RSVQA (free-form / open answers). |
| **VRSBench — captioning** | caption generation | ❌ INCOMPATIBLE | — | Production captioner is retrieval/concept composition, **not** a generative captioner; cannot produce valid predictions for BLEU/CIDEr/etc. Not misrepresented as generative. |
| **VRSBench — grounding** | referring-expression localization | ❌ BLOCKED (compatible, not run) | — | OWL-ViT grounding is architecturally compatible (box IoU@0.5); referring annotations downloaded and format confirmed (`referring.json`: expression + normalized bbox + image_id). Blocked only on the val image set (`Images_val.zip` ≈ 3.98 GB) not provisioned in this environment. Reproducible command provided. |

## Comparison points (like-for-like only)
- Change detection IoU/F1 compared across LEVIR-CD, OSCD, and the in-domain SECOND (Phase 9,
  IoU 0.4219 / F1 0.5934). **Never** compared against CDVQA accuracy or any VQA metric.
- The Phase 9 CDVQA classifier (Test 0.7114) and the Phase 10 external BLIP model
  (Test 0.5467, n=600) are VQA-accuracy numbers and are kept separate from all IoU/F1 above.

## Data / leakage (all benchmarks)
- LEVIR-CD, OSCD: **test split only** was used. No training, no threshold tuning, no model
  selection on these sets. The learned model is the frozen Phase 9 SECOND checkpoint; the Otsu
  reference is unsupervised per-image. → zero test-set leakage.
- No hidden ISRO/SAC data used anywhere.
- Datasets are public HF mirrors; only small test parquets were downloaded (LEVIR 69.5 MB, OSCD
  1.49 MB) — stored outside the repo, not committed.
