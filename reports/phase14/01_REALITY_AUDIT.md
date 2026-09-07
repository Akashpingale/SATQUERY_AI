# Phase 14 — Forensic Reality Audit

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07
**Method:** traced each query type through the actual code (`agent/understanding.py` → `agent/planner.py` → `agent/specialists/*` → `ml/inference/*`) and the UI evidence renderer (`frontend/src/components/results/EvidencePanel.tsx`). No production code was modified for this audit.

**Central question asked of every path:** *Is the system actually using a trained model to answer this, or producing an answer that merely looks intelligent?*

---

## 0. Environment reality (decisive for what Phase 14 can build)

| Resource | Reality | Consequence |
|---|---|---|
| GPU | **NVIDIA RTX 2050, 4 GB VRAM** (torch 2.6.0+cu124, CUDA available) | Only small/parameter-efficient training (LoRA, frozen encoders, cached features, tiny batches). Full generative-VLM or large VQA training is not realistic here |
| RAM | 16 GB | OK for CPU inference / small training |
| Network | HuggingFace reachable but **~0.02 MB/s** (measured: 1 MB / 47 s); other hosts fail DNS (`getaddrinfo failed`) | Multi-GB dataset downloads are **infeasible** (a 3.79 GB file ≈ 50 h). This already blocked VRSBench images |
| Local datasets | SECOND 2.3 GB (11,872 files) ✓; LEVIR-CD parquet ✓; OSCD parquet ✓; VRSBench annotations only (no images) | Only **temporal-change** work has usable local training data. No local RSVQA/RSICD/VRSBench-VQA/caption/segmentation/detection data |

**Implication:** the "train a real VQA / generative captioner / land-cover segmenter / detector" objectives (14A–14D) are **BLOCKED on dataset provisioning** in this environment, not on engineering. Temporal-change improvement (14F) is the one genuinely feasible real-training track.

---

## 1. Per-query forensic traces

Legend for the honesty verdict of each path: **REAL-TRAINED** (trained checkpoint), **REAL-DETERMINISTIC** (genuine algorithm on real pixels), **REAL-RETRIEVAL** (real model forward pass, retrieval/similarity — not generative/classifier), **REFUSED** (honest no-guess).

### Q1 "Describe this scene." → CAPTIONING
- Route: caption terms → `CaptioningSpecialist` → `clip_backend.score_concepts` (clip-rsicd-v2, +LoRA if present).
- Mechanism: image → CLIP image embedding → cosine similarity × logit_scale vs template-averaged concept-text embeddings → **softmax over the concept vocabulary** → top concepts composed into a sentence.
- Output honesty: answer + warning "composed from concept scores (retrieval-based), not generated free text"; evidence `method: "retrieval-based composition from zero-shot concept scores"`.
- **Verdict: REAL-RETRIEVAL. NOT generative captioning.** Correctly labeled. ✔ honest.

### Q2 "What land cover dominates this image?" → VQA (DOMINANT)
- Route: question terms → `VQASpecialist` → `analyze_question` → `QuestionKind.DOMINANT` → `clip_backend.score_land_cover` (land-cover concept subset).
- Mechanism: same CLIP similarity → softmax, restricted to land-cover concepts; answer = top concept + probability.
- Output honesty: "The scene is most consistent with {label} ({p:.0%})"; evidence note "values are relative to that vocabulary and are not calibrated probabilities of ground truth."
- **Verdict: REAL-RETRIEVAL.** It is a similarity score, **not** a pixel-area percentage and **not** a trained land-cover classifier. Correctly labeled. ✔ honest. ⚠ but see §2 (bar could still be misread).

### Q3 "Where is water visible?" → GROUNDING
- Route: grounding terms ("where is") + target extraction → `GroundingSpecialist` → OWL-ViT (`google/owlvit-base-patch32`); coarse CLIP patch-localization fallback for diffuse land cover.
- Mechanism: open-vocabulary detector real forward pass → boxes/overlay; or tile-contrast region (labeled coarse).
- **Verdict: REAL (OWL-ViT).** Functionally real; **public-benchmark IoU unmeasured** (VRSBench images blocked).

### Q4 "How many buildings are visible?" → VQA (COUNT) → REFUSED
- Route: "how many" → `analyze_question` → `QuestionKind.COUNT` → **refused**: "asks for a count, which the current model cannot provide … a contrastive image-text model has no counting mechanism."
- **Verdict: REFUSED (honest no-guess).** ✔ No fabricated number. There is **no** detector/counting model.

### Q5 "What changed between these images?" → TEMPORAL_CHANGE / CHANGE_VQA (deterministic)
- Route: change terms + T1/T2 → `registration_validator` → `change_vqa`/`temporal_change` (deterministic change-vector analysis `cva-robust-1.1.0`).
- Mechanism: co-registration + robust CVA on real pixels → changed-fraction, region, NDVI/NDWI direction.
- **Verdict: REAL-DETERMINISTIC** (default). Learned SECOND is **non-default** (see Q6/registry).

### Q6 "What changed from image A to image B?" (from-to / semantic) → CHANGE_VQA (deterministic)
- Mechanism: reports occurrence/location/extent/brightness direction + optional zero-shot land-cover *reading* before/after; **explicitly declines to name a from-to class** unless the reading supports it.
- **Verdict: REAL-DETERMINISTIC, binary/measurement-grounded — NOT semantic from-to classification.** Correctly labeled (no "semantic change accuracy" claim).

### Q7 "Compare optical and SAR." → OPTICAL_SAR_FUSION (deterministic)
- Route: fusion/cross-modal terms + optical+SAR pair → `registration_validator` → `optical_sar_fusion` (`xmodal-1.0.0`).
- Mechanism: co-registration + real feature-level cross-modal analysis (correlation, complementarity, per-modality structure, SAR-dark fraction) on real pixels.
- **Verdict: REAL-DETERMINISTIC feature-level fusion.** Genuine (not concatenated text). **Not** benchmarked vs optical-only/SAR-only on a labeled task.

### Q8 Grounding question (explicit target) → GROUNDING
- As Q3. **Verdict: REAL (OWL-ViT), IoU unproven.**

### Q9 CDVQA-style change question → CHANGE_VQA (deterministic default); learned `cdvqa_trained` exists but **non-default**
- The trained CDVQA classifier (CLIP features + GRU + fusion head; Test 0.7114) is registered and invocable but **not on any default toolchain** — the agent routes change-VQA to the deterministic specialist. `cdvqa_trained` reads RGB via PIL (SECOND-style aerial), cannot open multi-band GeoTIFF.
- **Verdict: deterministic REAL by default; learned classifier REAL-TRAINED but NOT_INTEGRATED into default routing.**

---

## 2. Critical-issue findings (from the prompt)

1. **"Is captioning generative?"** — **No, and it does not claim to be.** It is retrieval/concept composition, labeled as such in the answer, warning, and evidence `method`. **No fix required** (though a genuine generative path is the P1 improvement — data-blocked here).
2. **"Is the 56% presented as pixel-area?"** — **No.** The UI section is titled **"Model scores"**; each bar shows the concept probability with the note *"Softmax over the model's concept vocabulary; values are relative to that vocabulary and are not calibrated probabilities of ground truth."* It is **not** rendered as "56% of the image is industrial buildings." **Honesty is intact.** ⚠ Minor risk: a proportional bar can still be *visually* misread as area; a one-word label ("similarity") on the bar would remove all ambiguity — low-priority polish (P2), not a correctness defect.
3. **"Is a classifier called a VQA model / retrieval called generative / binary change called semantic?"** — **None of these mislabels exist.** The registry, specialists, and UI consistently use honest terminology.

**No anti-cheating violations were found:** no hardcoded answers, no filename/ground-truth lookup at inference, no fake probabilities (softmax over a real forward pass is disclosed as vocabulary-relative), no CLIP-as-generative, no binary-as-semantic, no test-set tuning in the shipped code.

---

## 3. What is genuinely trained vs not (summary)

| Capability | Backing | Honesty state |
|---|---|---|
| Single-image VQA | CLIP zero-shot concept scoring | REAL-RETRIEVAL (not trained VQA, not generative); counting REFUSED |
| Captioning | CLIP concept composition | REAL-RETRIEVAL (not generative) |
| Land cover | CLIP similarity (land-cover concepts) | REAL-RETRIEVAL (not segmentation; no area %) |
| Counting | — | REFUSED (no model) |
| Grounding | OWL-ViT (real) | REAL; IoU UNPROVEN (VRSBench blocked) |
| Temporal change | deterministic CVA (default) + learned SECOND (non-default, IoU 0.4219 in-domain) | REAL-DETERMINISTIC default; REAL-TRAINED non-default |
| Change VQA | deterministic (default) + CDVQA classifier (non-default, Test 0.7114) | REAL-DETERMINISTIC default; REAL-TRAINED NOT_INTEGRATED |
| Semantic (from-to) change | — (deterministic declines) | Correctly NOT claimed |
| Optical+SAR fusion | deterministic feature-level | REAL-DETERMINISTIC; not benchmarked vs single-modality |
| Agentic orchestration | keyword+shape routing, registry allowlist, validation, trace | REAL |

The system is **honest**: where it uses a trained model it says so; where it uses retrieval/deterministic/refusal it says so. The gap vs the SIH ideal is **capability depth** (no trained generative/VQA/segmentation/detection models), not integrity.
