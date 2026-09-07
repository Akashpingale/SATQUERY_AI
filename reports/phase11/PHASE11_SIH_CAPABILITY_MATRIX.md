# Phase 11 — SIH26167 Capability Matrix

Status vocabulary: **PASS** (real public-benchmark evidence), **PARTIAL** (real evidence but
limited/subset/in-domain-only), **NOT DEMONSTRATED**, ❌ **INCOMPATIBLE**, ❌ **BLOCKED**.
PASS is used only where an actual public evaluation supports it.

| SIH Capability | Current Implementation | Public Evidence | Benchmark | Status |
|---|---|---|---|---|
| Single-image VQA | Zero-shot CLIP concept scoring (`vqa`/`vqa_adapted`) | None (answer-format mismatch) | RSVQA, VRSBench-VQA | ❌ INCOMPATIBLE — cannot emit yes/no/count/area answers; needs a VQA head |
| Captioning | Retrieval/concept composition (`captioning`) | None (not generative) | VRSBench-Cap | ❌ INCOMPATIBLE — not a generative captioner |
| Grounding | OWL-ViT zero-shot (`grounding`) | Format confirmed; images not provisioned | VRSBench-referring | ❌ BLOCKED — compatible (box IoU) but ~4 GB val images not provisioned |
| Bi-temporal change (learned) | Siamese U-Net `second_change_learned` | **LEVIR-CD** IoU 0.077 / F1 0.144; **OSCD** IoU 0.105 / F1 0.191 (full test) | LEVIR-CD, OSCD | PARTIAL — real cross-dataset results, but poor generalization off SECOND (honest) |
| Bi-temporal change (deterministic) | CVA / Otsu radiometric (Phase 6) | **LEVIR-CD** IoU 0.047; **OSCD** IoU 0.145 / F1 0.254 (full test) | LEVIR-CD, OSCD | PARTIAL — robust reference; **beats learned model on OSCD** |
| Bi-temporal change (in-domain) | `second_change_learned` on SECOND | SECOND held-out IoU 0.4219 / F1 0.5934 (Phase 9) | SECOND | PASS (in-domain) |
| Change-based VQA | `cdvqa_trained` classifier | CDVQA Test 0.7114 / Test2 0.6632 (Phase 9) | CDVQA | PASS |
| Optical + SAR fusion | Feature-level `analyze_cross_modal` (Phase 7) | Synthetic property validation (Phase 9); no public benchmark exists | project eval | PARTIAL — validated, no matching public benchmark |
| Agentic orchestration | `run_analysis` planner + registry | Backend + Playwright E2E (Phases 3–9) | system tests | PASS |
| Evidence / audit trace | Trace + evidence + confidence | System tests; no chain-of-thought | system tests | PASS |
| BigEarthNet adaptation | Phase 5 LoRA CLIP | Held-out retrieval 0.0115→0.0791 mean recall (~6.9×) | BigEarthNet.txt | PASS (retrieval metric) |

## Reading of the change-detection evidence (honest)
- The learned SECOND detector is strong **in-domain** (SECOND IoU 0.42) but **does not generalize
  well** to LEVIR-CD (building-only labels → over-prediction, IoU 0.077) or OSCD (Sentinel-2
  sensor shift, IoU 0.105). On OSCD the simple deterministic method is actually better. This is
  a genuine, measured limitation — not hidden.
- Consequence for production: the deterministic detector remaining the robust default (project
  policy) is empirically supported; the learned model is a useful in-domain specialist, not a
  general change detector.
