# Phase 12 — Model / Specialist Capability Matrix

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-06
**Source of truth:** live `GET /api/v1/models` (14 registered tools; raw dump `reports/phase12/PHASE12_MODELS_REGISTRY_RAW.json`) cross-checked against the specialist code and the Phase 5/9/10/11 evidence.

This matrix describes **what each registered specialist actually does**, distinguishing generative VQA from retrieval/concept scoring, grounding, binary change detection, semantic change understanding, change-VQA, and optical-SAR fusion. Nothing here is inferred from a class name alone; each row was checked against its implementation and, where a number is quoted, against a committed results file.

> **On "status".** The registry health flag reports ML specialists as `degraded` and pure-algorithmic/utility tools as `available`/`planned`. `degraded` is an **honesty flag**, not a fault: it marks components whose scores are uncalibrated and/or whose configured device (some entries declare `cuda`) differs from the local CPU runtime. Every specialist below executed successfully on CPU during Phase 12 end-to-end validation.

---

## 1. Capability table

| Specialist (id) | Task | Input type | Training / adaptation source | Public benchmark evidence | Default? | Key limitations |
|---|---|---|---|---|---|---|
| **VQA baseline** (`vqa`) | Single-image VQA | Optical/SAR, 1 image | None — zero-shot use of upstream `flax-community/clip-rsicd-v2` | Not benchmarked as VQA | Fallback (used if no adapter) | **Retrieval/concept scoring, NOT generative.** Uncalibrated; no counting/spatial reasoning |
| **VQA adapted** (`vqa_adapted`) | Single-image VQA | Optical/SAR, 1 image | **LoRA** on BigEarthNet.txt (Lithuania/Summer Sentinel-2 captions), InfoNCE | BigEarthNet.txt held-out **retrieval R@ improved 0.0115→0.0791 (~6.9×)**, `reports/phase5` | **Default for VQA** (adapter present) | Adaptation is embedding-space alignment; improves retrieval, **does not add text generation or prove VQA accuracy**. Region/season/sensor benefit unestablished |
| **Captioning baseline** (`captioning`) | Captioning | Optical/SAR, 1 image | None (upstream CLIP) | Not benchmarked | Fallback | **Retrieval-composed, NOT generative** |
| **Captioning adapted** (`captioning_adapted`) | Captioning | Optical/SAR, 1 image | LoRA (same as `vqa_adapted`) | Phase 5 retrieval (as above) | Default for captioning | **Still retrieval-composed, NOT generative** — adaptation improved the embedding space only |
| **Grounding** (`grounding`) | Referring localization | Optical/SAR, 1 image | None — `google/owlvit-base-patch32` (Apache-2.0) | **Not benchmarked** (VRSBench image eval BLOCKED) | Default for grounding | Open-vocabulary boxes; IoU on a public split not yet measured (see grounding blocker) |
| **TemporalChange** (`temporal_change`) | Binary/radiometric change detection | Two co-registered images, same modality | None — deterministic **change vector analysis** | External change-detection context: LEVIR-CD / OSCD (see learned rows) | **Default for temporal change** | Detects *that* change occurred, not the semantic class; refuses cross-modality; needs band roles for NDVI/NDWI |
| **ChangeVQA** (`change_vqa`) | Change-VQA (measurement-grounded) | Two images + question | None — answers derived from CVA measurements | Not benchmarked (CDVQA is the learned row) | **Default for change-VQA** | Answers only occurrence/location/extent/NDVI-NDWI direction; never confirms "construction"; no urban classifier |
| **CDVQA trained** (`cdvqa_trained`) | Change-VQA (learned classifier) | Two optical images + question | Frozen CLIP image features + trained GRU + fusion head; **CDVQA questions + SECOND imagery** (train split) | **CDVQA held-out Test 0.7114 / Test2 0.6632** (majority 0.5084/0.4560), `reports/benchmarks/cdvqa` | **Non-default** (registered, invocable) | Closed 19-answer vocabulary; trained on 0.1524 m aerial RGB — other sensors/resolutions unproven; weak on counting; uncalibrated softmax; reads RGB via PIL |
| **SECOND learned** (`second_change_learned`) | Binary change detection (learned) | Two optical images | **Siamese U-Net from scratch**, SECOND train split (binary from semantic labels), Dice+BCE | In-domain held-out **IoU 0.4219 / F1 0.5934**; **cross-dataset LEVIR-CD (n=2048) IoU 0.0773 / F1 0.1435**; **OSCD (n=50) IoU 0.1053 / F1 0.1905** — `reports/phase11` | **Non-default** (registered, invocable) | Binary change only; strong in-domain, **weak cross-sensor**; on OSCD the deterministic Otsu reference (IoU 0.1453) **beats** it; reads RGB via PIL |
| **Optical-SAR fusion** (`optical_sar_fusion`) | Optical+SAR complementary analysis | One optical + one SAR, co-registered | None — deterministic **feature-level cross-modal** analysis | Not benchmarked | Default for fusion | Requires exactly one optical + one SAR; SAR treated as uncalibrated log-backscatter; reports backscatter/roughness, not semantic classes |
| **RasterPreprocessor** (`raster_preprocessor`) | Metadata/preview | Any raster | N/A (Rasterio/GDAL) | N/A | Default (ingestion) | — |
| **RegistrationValidator** (`registration_validator`) | Pair compatibility | Two rasters | N/A (Rasterio/pyproj) | N/A | Default (guards every pair) | — |
| **EvidenceAggregator / ConfidenceCalibrator** | Utility | — | N/A | N/A | `planned` (logic lives inline in the pipeline) | Placeholder registry entries; the actual evidence/confidence assembly runs in the orchestrator |
| **External `change_model`** (BLIP) | Change-VQA (external) | Single-image BLIP VQA | Externally supplied checkpoint (361M params, ~1.35 GB) | **CDVQA Test 0.5467 (n=600)** — `reports/change_model` | **NOT INTEGRATED** | Phase 10 verdict unchanged: `VERIFIED_BUT_NOT_INTEGRATED`. Scored **below** the project's `cdvqa_trained` (0.7114); single-image architecture. Preserved, gitignored, not on any route |

---

## 2. Capability-type distinctions (explicitly preserved)

- **Generative VQA:** *Not claimed.* The VQA specialists are **CLIP concept scoring / retrieval**, not text generation. The learned `cdvqa_trained` is a **closed-vocabulary classifier**, also not free-form generation.
- **Retrieval / concept scoring:** `vqa`, `vqa_adapted`, `captioning`, `captioning_adapted`. Captioning is retrieval composition, never presented as a generative captioner.
- **Grounding:** `grounding` (OWL-ViT) — real open-vocabulary box localization; public-split IoU not yet measured (BLOCKED).
- **Binary change detection:** deterministic `temporal_change` (default) and learned `second_change_learned` (non-default).
- **Semantic change understanding:** *Not claimed as a from-to classifier.* The learned SECOND model is **binary** change (changed vs unchanged); the deterministic route adds an optional zero-shot land-cover *reading* that never decides whether change occurred.
- **Change-VQA:** deterministic measurement-grounded `change_vqa` (default) and learned `cdvqa_trained` (non-default, benchmarked).
- **Optical-SAR fusion:** genuine feature-level cross-modal analysis (`optical_sar_fusion`), not concatenated text.

---

## 3. Adaptation & benchmark claims — bounded

| Claim | What is proven | What is NOT proven |
|---|---|---|
| BigEarthNet LoRA adaptation | Held-out **retrieval** improvement ~6.9× (Phase 5) — genuine RS adaptation wired into the serving VQA/captioning route | Does **not** prove VQA or captioning *accuracy*; benefit off Lithuania/Summer Sentinel-2 unestablished |
| CDVQA `cdvqa_trained` | **Test 0.7114 / Test2 0.6632** on the held-out CDVQA benchmark, above majority baseline | Does **not** generalize to arbitrary sensors/resolutions; closed 19-answer vocabulary |
| SECOND `second_change_learned` | **In-domain IoU 0.4219 / F1 0.5934** — strong learned change detection on SECOND | **Cross-sensor generalization is weak** (LEVIR-CD/OSCD); deterministic wins on OSCD |
| LEVIR-CD | External public change-detection evaluation **completed** (n=2048) | — |
| OSCD | External public change-detection evaluation **completed** (n=50); **deterministic method currently outperforms the learned SECOND model** | — |
| External `change_model` | Independently **verified** at CDVQA Test 0.5467 (n=600) | Below the project's own model; remains **NOT INTEGRATED** |
