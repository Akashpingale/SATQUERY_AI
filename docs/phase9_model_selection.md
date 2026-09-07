# Phase 9 Model Selection

Selection is driven by measured evidence. Where no new model was trained (external datasets not
provisioned in this environment), the incumbent production model is retained and that is stated
plainly. No model is promoted to `available` merely because code exists.

## Single-image VQA
- **SELECTED (production):** `vqa` baseline (zero-shot CLIP concept scoring), with `vqa_adapted`
  (Phase 5 LoRA) preferred automatically when the adapter is present.
- **BASELINE:** same. **VALIDATION/TEST:** none on RSVQA/VRSBench (NOT RUN — see below).
- **REASON:** no VQA-answer-format model was trained. RSVQA/VRSBench VQA answers (yes/no, counts,
  areas) are incompatible with concept scoring, so no valid benchmark selection could be made.
- **LIMITATIONS:** remains `degraded`; a constrained/generative VQA head is required for a real
  RSVQA/VRSBench run.

## Captioning
- **SELECTED:** retrieval-composition captioner (baseline/adapted). **REASON:** unchanged; no
  generative captioner trained. **LIMITATIONS:** not benchmark-compatible with generative caption
  metrics; remains `degraded`.

## Grounding
- **SELECTED:** OWL-ViT zero-shot. **REASON:** unchanged; VRSBench not provisioned, so no
  grounding fine-tuning/selection performed. **LIMITATIONS:** zero-shot; remains `degraded`.

## Adapted VLM (retrieval)
- **SELECTED:** Phase 5 LoRA adapter (`clip-rsicd-v2 + lora-ben-txt-r16a32`), already integrated
  and auto-preferred. **VALIDATION/TEST:** BigEarthNet.txt held-out test, mean recall
  0.0115 → 0.0791. **REASON:** large, measured retrieval improvement on held-out data.
  **LIMITATIONS:** retrieval metric only; Lithuania/Summer subset; uncalibrated.

## Change detection / Change VQA
- **SELECTED (production default):** Phase 6 deterministic `cva-robust`. **REASON:** robust,
  reproducible, needs no weights/GPU; correctly localizes synthetic changes and is specific on
  no-change. No learned detector was trained (SECOND/LEVIR-CD/OSCD/CDVQA not provisioned), so no
  learned model could be selected. **LIMITATIONS:** radiometric (not semantic) change; remains
  `degraded`. A future learned detector would be added as a selectable specialist, not a
  replacement.

## Optical + SAR fusion
- **SELECTED:** Phase 7 feature-level `analyze_cross_modal`. **REASON:** unchanged; no public
  benchmark to select against. **LIMITATIONS:** uncalibrated SAR; remains `degraded`.

## Confidence
- **SELECTED:** existing `app.agent.confidence.assess` with `HIGH≥0.75`, `MEDIUM≥0.45`.
- **REASON:** calibration was measured (synthetic, N=12) but is insufficient to justify changing
  thresholds; the system is mildly under-confident on that fixture. Changing thresholds requires
  real-benchmark validation-split calibration (future work). **Thresholds unchanged.**

## Phase 9 (datasets provisioned) — trained models selected on validation

Two learned models were trained on real data and selected on validation, then evaluated once on
held-out test:

- **CDVQA — `cdvqa_trained`.** SELECTED via best **Val** accuracy (0.7166 at epoch 15/20).
  Held-out **Test 0.7114 / Test2 0.6632** vs majority baseline 0.5084/0.4560. Calibrated
  (Val→Test temperature scaling, ECE 0.014). Latency: negligible after one-time CLIP feature
  extraction. REASON: large, measured accuracy gain over baseline on disjoint held-out data.
  LIMITATIONS: domain-specific (SECOND 0.15 m aerial), closed 19-answer vocab, weak on
  counting/ratio. **Registered `degraded`; NOT the default route** — the deterministic detector
  is kept for general imagery (generalization/robustness).
- **SECOND — `second_change_learned`.** SELECTED via best **Val IoU** (0.42). Held-out **IoU
  0.4219 / F1 0.5934** vs deterministic reference 0.2248 / 0.3670 (~2×). REASON: clear held-out
  improvement over the deterministic baseline. LIMITATIONS: binary change only, aerial RGB,
  internal held-out split. **Registered `degraded`; NOT the default route** — deterministic CVA
  remains default for general/georeferenced imagery.

## Net registry outcome
The two learned specialists (`cdvqa_trained`, `second_change_learned`) were added, benchmarked,
integrated, and verified — but kept at `degraded` and non-default, because each is validated only
on a single domain-specific dataset while the deterministic detectors remain the more general
production default. This is the honest, non-inflated outcome: promotion to `available` awaits
multi-dataset / cross-sensor validation.
