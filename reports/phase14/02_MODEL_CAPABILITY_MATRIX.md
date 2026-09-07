# Phase 14 — Current Model Capability Matrix

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07
**Source:** live registry (`GET /api/v1/models`, 14 tools) + code trace + Phase 5/9/11 evidence. States use the Phase 14 ladder: IMPLEMENTED · TRAINED · VALIDATED · BENCHMARKED · VERIFIED · PRODUCTION_INTEGRATED · PRODUCTION_VERIFIED.

| Capability | Current implementation | Actual architecture | Checkpoint | Training status | Validation | Benchmark (metric) | Production status | Key limitation |
|---|---|---|---|---|---|---|---|---|
| **Single-image VQA** | CLIP zero-shot concept scoring (`vqa`/`vqa_adapted`) | Contrastive CLIP (clip-rsicd-v2) + optional LoRA | `flax-community/clip-rsicd-v2` (+`models/phase5/adapter`) | Not trained as VQA; LoRA adapts embeddings | Phase 5 retrieval only | Not benchmarked as VQA | **PRODUCTION_INTEGRATED (retrieval)** | **Retrieval, not generative/classifier VQA**; no counting/spatial reasoning; uncalibrated |
| **Captioning** | Concept-score composition (`captioning`/`captioning_adapted`) | Contrastive CLIP + composer | same as above | Not trained (composition) | Phase 5 retrieval | Not benchmarked (no caption metrics) | **PRODUCTION_INTEGRATED (retrieval)** | **Not generative**; no BLEU/CIDEr (not applicable to retrieval) |
| **Land-cover understanding** | CLIP similarity over land-cover concepts (VQA DOMINANT) | Contrastive CLIP | same | Not trained | — | Not benchmarked | **PRODUCTION_INTEGRATED (retrieval)** | **Similarity score, not segmentation**; no pixel-area; no trained classifier |
| **Grounding** | OWL-ViT open-vocab + CLIP patch fallback (`grounding`) | OWL-ViT (owlvit-base-patch32) | `google/owlvit-base-patch32` | Pretrained (not fine-tuned) | — | **UNPROVEN** (VRSBench IoU blocked) | **PRODUCTION_INTEGRATED**, benchmark UNPROVEN | Public-split IoU unmeasured; open-vocab, coarse for diffuse cover |
| **Counting** | Refused | — | — | — | — | — | **Refuses (no-guess)** | No detector/counting model exists |
| **Temporal change (binary)** | Deterministic CVA default; learned SECOND non-default | Robust change-vector analysis; Siamese U-Net | none / `models/phase9/second/second_best.pt` | CVA none; SECOND trained | SECOND internal val | **BENCHMARKED**: SECOND in-domain IoU 0.4219/F1 0.5934; LEVIR-CD n=2048 IoU 0.0773; OSCD n=50 IoU 0.1053 (Otsu 0.1453 wins) | Deterministic **PRODUCTION_VERIFIED**; SECOND **VALIDATED, non-default** | Learned cross-sensor **weak**; CVA is radiometric change, not class |
| **Change VQA** | Deterministic measurement default; CDVQA classifier non-default | CVA-grounded answers; CLIP+GRU+fusion head | none / `models/phase9/cdvqa/cdvqa_best.pt` | CDVQA trained | CDVQA held-out | **BENCHMARKED**: CDVQA Test 0.7114 / Test2 0.6632 (majority 0.5084/0.4560) | Deterministic **PRODUCTION_VERIFIED**; CDVQA **VALIDATED, NOT_INTEGRATED** | Closed 19-answer vocab; SECOND-domain only; reads RGB PNG (not GeoTIFF) |
| **Semantic (from-to) change** | Not implemented (deterministic declines) | — | — | — | — | — | **Not claimed** | Would need a from-to classifier + labeled data |
| **Optical+SAR fusion** | Deterministic feature-level cross-modal (`optical_sar_fusion`) | NumPy/SciPy cross-modal analysis | none | none (algorithmic) | — | Not benchmarked vs single-modality | **PRODUCTION_VERIFIED (deterministic)** | Genuine fusion but no accuracy benchmark vs optical-only/SAR-only |
| **Agentic orchestration** | Keyword+shape routing, registry allowlist, plan validation, trace | Rule-based planner | — | — | — | E2E flows A–E | **PRODUCTION_VERIFIED** | Rule-based (not an LLM planner) — deliberate for auditability |
| **External change_model (BLIP)** | Not integrated | BlipForQuestionAnswering (361M) | `change_model/` (gitignored) | External | — | CDVQA Test 0.5467 (n=600) | **VERIFIED_BUT_NOT_INTEGRATED** | Below project's own CDVQA (0.7114); single-image |

## Reading of the matrix

- **PRODUCTION_VERIFIED (real trained/deterministic, benchmarked, integrated, verified):** deterministic temporal change, deterministic change-VQA, deterministic optical-SAR fusion, agentic orchestration.
- **PRODUCTION_INTEGRATED but retrieval (not trained for the task):** VQA, captioning, land-cover — real forward passes, honestly labeled, but not trained generative/classifier/segmentation models.
- **VALIDATED / BENCHMARKED but NON-DEFAULT:** learned SECOND (temporal), CDVQA classifier (change-VQA).
- **UNPROVEN:** grounding IoU.
- **NOT_INTEGRATED:** external BLIP `change_model`.
- **Not implemented (and not claimed):** trained VQA head, generative captioner, land-cover segmentation, object detector/counting, semantic from-to change.
