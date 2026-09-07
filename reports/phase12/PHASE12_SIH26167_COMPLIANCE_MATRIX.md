# Phase 12 — SIH26167 Compliance Matrix

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-06
**Statuses:** `PASS` · `PARTIAL` · `UNPROVEN` · `INCOMPATIBLE` · `BLOCKED` · `NOT APPLICABLE`

Rules honored: **no `UNPROVEN`/`PARTIAL` was promoted to `PASS`**; a component is not called "generative" merely because it emits text; benchmark claims are bounded to what was actually measured. Evidence points to live traces (`reports/phase12/e2e/`), the registry (`GET /api/v1/models`), committed benchmark files, or code paths.

| # | Requirement | Status | Evidence (location) | Limitation / missing evidence |
|---|---|---|---|---|
| 1 | Single-image optical/multispectral analysis | **PASS** | Flow A live trace `e2e/flow_a_single_image.json`; `raster_preprocessor`/ingestion | Answer scores are uncalibrated (zero-shot CLIP) |
| 2 | Single-image SAR analysis | **PARTIAL** | SAR ingests + previews + is analyzed within fusion (Flow C `e2e/flow_c_optical_sar.json`); `demo_sar.tif` | No SAR-specific *semantic* classifier; SAR VQA is the same zero-shot CLIP concept scoring, not SAR-calibrated |
| 3 | Single-image VQA | **PASS (bounded)** | Flow A; `vqa_adapted` route | **Retrieval/concept scoring, not generative**; no counting/spatial reasoning; not benchmarked on a public VQA set (see #18) |
| 4 | Captioning | **PASS (bounded)** | `captioning_adapted` registered + routed; component tests | **Retrieval-composed, NOT generative**; generative caption metrics N/A |
| 5 | Grounding | **PARTIAL** | `grounding` (OWL-ViT) wired + demo flow; box overlays render | Public-split **IoU not measured** — VRSBench image eval is **BLOCKED** (see #5 detail + row 20 of tasks). Functionally works; quantitatively unproven |
| 6 | Bi-temporal analysis | **PASS** | Flow B live trace `e2e/flow_b_temporal.json` (co-registration + registration check) | — |
| 7 | Change detection | **PASS** | Flow B (deterministic CVA, 8.6% change, coherent region); learned SECOND in-domain IoU 0.4219 (`reports/benchmarks/second`) | Learned cross-sensor weak (LEVIR-CD/OSCD, `reports/phase11`); deterministic is default |
| 8 | Change description | **PASS** | Flow B answer describes extent/location/direction from measurements | Describes *that*/where/how much; **does not classify the change type** (no urban classifier) |
| 9 | Change-based VQA | **PASS** | Deterministic `change_vqa` (Flow B); learned `cdvqa_trained` **Test 0.7114/Test2 0.6632** (`reports/benchmarks/cdvqa`); Flow D | Deterministic answers only measurement-supported questions; learned is closed-vocab, non-default |
| 10 | Optical+SAR complementary analysis | **PASS** | Flow C live trace: genuine feature-level cross-modal (correlation 0.823, complementarity 0.1766) | SAR uncalibrated log-backscatter; reports roughness, not semantic classes |
| 11 | Agentic orchestration | **PASS** | All flow traces; `PHASE12_END_TO_END_TRACE_EVIDENCE.md` | — |
| 12 | Validation / tool selection / execution | **PASS** | `validate_plan`; registry allowlist; Flow E rejects before execution | — |
| 13 | Result combination | **PASS** | Multi-step plans combine registration + specialist + evidence aggregation (Flows B/C, 11–15 evidence items) | — |
| 14 | Confidence | **PASS** | High/Medium/Low/Unable across flows with auditable signals (`confidence.signals`) | Underlying model scores uncalibrated (surfaced honestly) |
| 15 | Evidence | **PASS** | Typed evidence in every completed flow; `/evidence/{token}` image serving | — |
| 16 | Auditable trace | **PASS** | Observable step-by-step trace in every flow; no chain-of-thought | — |
| 17 | VLM adaptation / fine-tuning | **PASS (bounded)** | Phase 5 LoRA on BigEarthNet.txt wired into serving; retrieval **0.0115→0.0791 (~6.9×)** (`reports/phase5`) | Proves **adaptation/retrieval gain**, *not* VQA/caption accuracy; single region/season/sensor |
| 18 | Public benchmark evidence | **PARTIAL** | CDVQA (0.7114/0.6632), SECOND in-domain (IoU 0.4219), LEVIR-CD (n=2048), OSCD (n=50), BigEarthNet retrieval — all committed | RSVQA/VRSBench-VQA/captioning **INCOMPATIBLE** (format); VRSBench grounding **BLOCKED** |
| 19 | Hidden-data isolation | **PASS** | Only public datasets + synthetic demo fixtures (`data/manifests/demo_manifest.json`, `hidden_evaluation_data: false`); no ISRO/SAC data anywhere | — |
| 20 | GeoTIFF/TIFF workflow | **PASS** | All demo fixtures are GeoTIFF; full ingest→analyze in Flows A/B/C | — |
| 21 | PNG/JPEG benchmark workflow | **PASS** | Upload accepts PNG/JPEG (`is_allowed_content_type`); benchmarks (CDVQA/SECOND/LEVIR/OSCD) run on PNG | — |
| 22 | Failure / fallback behavior | **PASS** | Flow E: incompatible→rejected, unsupported→Unable to determine, compatibility endpoint→false; no fabrication | — |
| 23 | Reproducibility | **PASS** | Deterministic fixtures + committed manifests/checksums; documented commands; `scripts/phase12/*` | Some evals need external datasets on disk (documented paths) |
| 24 | Frontend demonstration | **PASS** | 126/126 vitest, Playwright demo flows, live Vite server; the six demo flows run on the real workflow | Human screen-reader (NVDA) verification **PENDING** (see accessibility) |

### Requirement 5 / 18 detail — INCOMPATIBLE vs BLOCKED (unchanged from Phase 11, re-verified)

- **RSVQA (VQA):** `INCOMPATIBLE` — production VQA is zero-shot CLIP concept scoring over a fixed vocabulary; it cannot emit RSVQA yes/no/count/area answers without a constrained/generative head.
- **VRSBench-VQA:** `INCOMPATIBLE` — same answer-format incompatibility.
- **VRSBench-Captioning:** `INCOMPATIBLE` — production captioner is retrieval/concept composition, not generative; cannot produce valid predictions for generative caption metrics.
- **VRSBench-Grounding:** `BLOCKED` — OWL-ViT is architecturally compatible and the referring annotations are present (`D:\dataset\vrsbench\referring.json`); blocked solely on the ~3.98 GB `Images_val.zip` not being provisioned. See `PHASE12_VRSBENCH_GROUNDING_BLOCKER.md`.

## Summary

- **PASS:** 18 requirements (incl. 4 bounded PASS where wording is carefully constrained).
- **PARTIAL:** 3 (single-image SAR semantics #2, grounding quantitative #5, public-benchmark breadth #18).
- **INCOMPATIBLE:** RSVQA, VRSBench-VQA, VRSBench-Captioning (format-level, documented).
- **BLOCKED:** VRSBench grounding IoU (missing image archive).
- **UNPROVEN / NOT APPLICABLE:** none outstanding beyond the above.

All mandatory SIH product capabilities (single-image VQA, a second single-image task, bi-temporal change, genuine optical-SAR fusion, RS VLM adaptation, and answer+evidence+confidence+trace on every result) are **implemented, executed, and demonstrated on the real workflow**.
