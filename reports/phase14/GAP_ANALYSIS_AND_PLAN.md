# Phase 14 — Gap Analysis & Prioritized Implementation Plan

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07
**Rule:** feasibility is graded against the measured environment (RTX 2050 **4 GB VRAM**; HF **~0.02 MB/s**; non-HF hosts DNS-restricted; local data = SECOND/LEVIR/OSCD parquet/VRSBench annotations). No training result is claimed unless actually produced.

---

## A. Gap analysis (SIH ideal vs current reality)

| # | Capability | Current | Gap to "real trained" | Blocking factor |
|---|---|---|---|---|
| 1 | Single-image VQA | CLIP retrieval | No trained VQA (classifier or generative) | **Dataset** (RSVQA/VRSBench-VQA not local; download infeasible) |
| 2 | Generative captioning | CLIP retrieval composition | No image→text decoder | **Dataset** (caption sets not local) + tight VRAM |
| 3 | Land-cover understanding | CLIP similarity | No segmentation / trained classifier; no pixel-area | **Dataset** (segmentation set not local) |
| 4 | Counting | Refused | No detector | **Dataset** (detection set not local) |
| 5 | Grounding | OWL-ViT (real) | No measured IoU | **Data** (VRSBench images 3.79 GB @ 0.02 MB/s) |
| 6 | Temporal change | deterministic + learned SECOND | Weak cross-sensor generalization | **Feasible to improve** (local SECOND) |
| 7 | Semantic from-to change | not built (declined) | No from-to classifier | Heavy on 4 GB VRAM; partial data (SECOND labels) |
| 8 | Optical+SAR fusion | deterministic feature-level | No accuracy benchmark vs single-modality | **Dataset** (labeled opt+SAR not local) |
| 9 | Confidence | signal-based, uncalibrated | No calibration (temperature/Platt) | Feasible small task |
| 10 | Orchestration | rule-based, verified | (none — deliberate) | — |

**Integrity gap: none.** The gaps are capability depth, not honesty. Nothing is mislabeled or fabricated (see `01_REALITY_AUDIT.md`).

---

## B. Feasibility verdict per Phase-14 sub-goal

| Sub-goal | Verdict | Why |
|---|---|---|
| 14A real VQA | **BLOCKED** | dataset provisioning infeasible here |
| 14B generative captioning | **BLOCKED** | dataset provisioning + 4 GB VRAM |
| 14C land-cover segmentation | **BLOCKED** | dataset provisioning |
| 14D detector/counting | **BLOCKED** | dataset provisioning |
| 14E grounding IoU | **BLOCKED** | VRSBench images ~50 h @ 0.02 MB/s |
| 14F temporal improvement | **FEASIBLE** | SECOND local (2.3 GB); small Siamese U-Net fits 4 GB |
| 14G semantic from-to change | **PARTIAL/RISKY** | SECOND labels local, but real model + eval heavy on 4 GB |
| 14H opt+SAR benchmark | **BLOCKED** | no labeled co-registered opt+SAR local |
| 14J confidence calibration | **FEASIBLE (small)** | can calibrate on existing held-out predictions |

---

## C. Prioritized implementation plan

### P0 — do not regress
Preserve every existing model/route (Phase 5 LoRA, deterministic CVA, SECOND, CDVQA, fusion, OWL-ViT, external `change_model`). Deterministic defaults stay default. No deletion, no overwrite. **This constrains all work below to be additive.**

### P1 — Feasible real work in THIS environment
1. **14F Temporal-change improvement (real training on local SECOND).**
   - Additive experiments vs the existing SECOND Siamese U-Net (kept as-is): augmentation, normalization, Dice+BCE/focal/boundary-aware loss, balanced sampling, multi-scale/pretrained encoder.
   - Protocol: train split only; select on val; **frozen** SECOND held-out test; re-run LEVIR-CD/OSCD cross-dataset (no threshold tuning on test). Retain per-experiment manifest (seed, hp, ckpt hash, metrics).
   - Only promote a new checkpoint if it **beats** the current one on the frozen protocol; otherwise keep as experimental and document. Deterministic default unchanged regardless.
2. **14J Confidence calibration (small).** Fit temperature/Platt scaling on existing held-out predictions for the learned models; report ECE before/after. No new data needed.
3. **Optional opt-in routing for the learned models** where inputs are provably compatible (SECOND-style aerial RGB), **without** changing the deterministic default — makes the already-trained CDVQA/SECOND reachable in a guarded way. Additive, low-risk.

### P2 — Low-risk polish
4. **UI clarity:** add a "similarity" qualifier to the concept-score bars (the note already says vocabulary-relative; this removes any residual area misreading). Cosmetic, no behavior change.

### BLOCKED — needs provisioning elsewhere (documented, not attempted here)
5. **14A/B/C/D/E/H:** require multi-GB public datasets (RSVQA, VRSBench VQA/Cap/images, segmentation, detection, labeled opt+SAR). Provision on a machine with normal bandwidth (or copy archives in out-of-band), then run the documented training/eval. Grounding procedure + fetch script already exist (`scripts/phase13/fetch_vrsbench_val.py`, Phase 13 blocker doc).

---

## D. Honest bottom line

In this environment, the only genuine "train a real model" track that can actually run is **14F (temporal change on local SECOND)** plus small **confidence calibration**. Everything else in 14A–14H is **BLOCKED on dataset provisioning** (network ~0.02 MB/s, non-HF DNS restricted) — and per the phase's own stop conditions, that is reported as BLOCKED rather than faked. No fabricated VQA/captioning/segmentation/detection models will be produced.
