# Phase 12 — End-to-End Trace Evidence

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-06
**How produced:** the running application was driven through its **public API** (`http://localhost:8000/api/v1`) with the committed synthetic demo fixtures (`data/demo/*.tif`). The learned non-default route was exercised directly on the public SECOND aerial PNGs. Full machine-readable dumps live in `reports/phase12/e2e/` (`flow_a_single_image.json`, `flow_b_temporal.json`, `flow_c_optical_sar.json`, `flow_e_failure_fallback.json`, `flow_d_learned.json`, `SUMMARY.json`).

**Reproduce:**
```powershell
# servers already running (uvicorn :8000, vite :5173)
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
$env:HF_HOME="D:\SatQuery-AI\models\hf-cache"; $env:DEVICE="cpu"
.\.venv\Scripts\python.exe scripts\phase12\e2e_validate.py       # Flows A,B,C,E (live API)
.\.venv\Scripts\python.exe scripts\phase12\flow_d_learned.py --pairs 3   # Flow D (learned)
```

Every flow was checked against the nine agentic-orchestration obligations: **query understanding → task decomposition → validation → specialist selection → execution → result combination → confidence → evidence → auditable trace.**

---

## Flow A — Single image (VQA)

- **Input:** `demo_optical.tif` (3-band optical, EPSG:32643, georeferenced).
- **Query:** "What land cover types are visible in this scene?"
- **Result:** `status=completed`, `task=vqa`.
- **Answer:** "The scene is most consistent with water (52%). The next closest match is a river (24%)."
- **Confidence:** **Medium (0.47)** — signals include result strength 0.52, routing score 0.43, all inputs georeferenced, 4 evidence items, 2 warnings reduced confidence.
- **Model selection:** routed to **`vqa_adapted`** (`clip-rsicd-v2+lora-ben-txt`) with the trace note *"selected the Phase 5 adapted model (adapter present)"*.
- **Evidence (4):** `scores`, `model`, `model`, `preview`.
- **Honesty:** warnings state the scores are **uncalibrated zero-shot** and that adaptation was trained on a Lithuania/Summer Sentinel-2 subset, so off-domain benefit is unestablished.
- **Orchestration proof:** trace exposes Input inspection → Query understanding → Task classification (`vqa`, routing 0.43, matched "visible","what") → Validation → Tool selection → Model selection → Model provenance → Limitations → Execution → Evidence → Confidence.

## Flow B — Bi-temporal change

- **Input:** `demo_optical_t1.tif` (t1) + `demo_optical_t2.tif` (t2, contains a synthetic development block in the NW).
- **Query:** "What changed between these two images?"
- **Result:** `status=completed`, `task=change_vqa`, routed to the **deterministic** toolchain `registration_validator` → `change_vqa` (`cva-robust-1.1.0`). **The learned models were NOT selected** — deterministic is the default.
- **Answer:** 100% spatial overlap; **change affects 8.6% of the shared area (~219,300 m²), concentrated in the northern part** (near 13.5644, 75.0053), brighter in the later image; a land-cover reading returns "water" both before and after, so the system reports *that* something changed **without inventing a cause**.
- **Confidence:** **High (0.867)** — result strength 0.94, 11 evidence items.
- **Evidence (11):** `comparison`, `preview`×4, `change_map`, `temporal_compare`, `change_region`, `indices`, `scores`, `model`.
- **Orchestration proof:** trace records co-registration (resampled both onto a shared 160×160 grid, 100% valid, EPSG:32643), a registration cross-correlation check (0.817 at offset (0,0)), and the CVA threshold selection (2193/25600 px changed, 1 coherent region). The detected region matches the injected change block — a real detection, not a canned string.

## Flow C — Optical + SAR fusion

- **Input:** `demo_optical.tif` (optical) + `demo_sar.tif` (SAR backscatter).
- **Query:** "Compare what the optical and SAR sensors reveal about this area."
- **Result:** `status=completed`, `task=optical_sar_fusion`, toolchain `registration_validator` → `optical_sar_fusion` (`xmodal-1.0.0`).
- **Answer:** SAR shows structure the optical image does not over **2.9%** of the scene; SAR treated as **log10 uncalibrated backscatter** (relative intensity only); structured fraction 7.9% (SAR) vs 5.0% (optical); 25% SAR-dark (smooth surfaces); named complementary regions with coordinates; notes SAR is illumination-independent.
- **Confidence:** **High (0.952)** — result strength 1.00, 15 evidence items.
- **Evidence (15):** includes `fusion_composite`, `fusion_complementary`, `aligned_optical`, `aligned_sar`, `modality_comparison`, `cross_modal_water`, `complementary_region`×2.
- **Orchestration proof:** trace records genuine **feature-level cross-modal analysis** (correlation 0.823, complementarity 0.1766, optical-only/SAR-only/shared structure split) — not two concatenated text outputs.

## Flow D — Learned temporal route (non-default)

Exercised directly because the agent **never auto-selects** it (see routing guard below).

- **Routing guard:** `learned_tools_on_default_toolchain = []`; `TASK_TOOLCHAIN` maps temporal/change-VQA to the deterministic specialists. **Deterministic remains the default — confirmed.**
- **Availability:** both checkpoints report `available` and load on CPU (`models/phase9/second/second_best.pt`, `models/phase9/cdvqa/cdvqa_best.pt`).
- **SECOND (`second_change_learned`) on 3 real SECOND pairs:** changed-fraction **0.496 / 0.480 / 0.340**, mean confidence 0.73–0.78, all `output_valid` (∈ [0,1]).
- **CDVQA (`cdvqa_trained`) on the same pairs, "Did new buildings appear?":** answers "yes" with softmax **0.569 / 0.777 / 0.871** over the closed 19-answer vocabulary.
- **Not silently forced:** the learned specialists read RGB via PIL and target SECOND-style aerial RGB. They **cannot open** the multi-band uint16 GeoTIFF demo fixtures (`PIL.UnidentifiedImageError`), and the agent does not route GeoTIFF pairs to them — the deterministic route handled Flow B instead. This is exactly why the deterministic detector remains the robust default.
- **Safety:** checkpoints loaded read-only; nothing retrained or overwritten.

## Flow E — Failure / fallback

- **E1 incompatible pair** (`demo_optical.tif` + `demo_incompatible.tif`, non-overlapping footprint), query "What changed…":
  - `status=rejected`, issue `incompatible_inputs` ("images do not overlap geographically").
  - **No fabricated answer.** Trace shows the workflow `failed`/stopped **after the compatibility check**, before any specialist executed. Two preview evidence items retained.
- **E2 unsupported query** ("Predict tomorrow's weather at this location.") on a valid image:
  - `status=rejected`, `task=unsupported`, `answer=null`, confidence **Unable to determine**.
  - Trace stops at Task classification with a helpful message; no specialist ran; nothing invented.
- **E3 compatibility endpoint** on the same non-overlapping pair: `compatible=false`, `overlap_ratio=0.0`, issue `no_overlap`.

---

## Agentic orchestration — verdict

Across the flows the trace consistently and observably exposes: **what the user asked** (query understanding), **what task was inferred and why** (classification with routing score + matched signals), **which specialist was selected and why** (tool + model-selection notes, adapter-vs-baseline), **what executed** (per-step execution entries with versions and parameters), **what evidence was produced** (typed evidence items), **confidence with its signals**, and **fallback/degraded behavior** (rejections, limitations, unavailability). No trace entry exposes private chain-of-thought — only factual, auditable operations. This satisfies the SIH agentic-orchestration requirement on the real production workflow.
