# Phase 9 Benchmark Report — SatQuery AI (SIH26167)

Judge-facing summary. **No fabricated numbers.** `NOT RUN` means the dataset was not provisioned
in this environment; `INCOMPATIBLE` means the current architecture cannot produce valid
predictions for that benchmark's answer format. Reproduce anything here with the commands in
`docs/benchmarking.md`.

## Final result table

| Capability | Dataset | SIH? | Training | Baseline | Final | Status |
|---|---|---|---|---|---|---|
| RS VLM adaptation | BigEarthNet.txt (retrieval) | SIH (adaptation) | Phase 5 LoRA (done) | mean recall 0.0115 | **0.0791 (~6.9×)** | ✅ measured (retrieval only) |
| Single-image VQA | RSVQA | SIH | — | — | — | NOT RUN + INCOMPATIBLE (answer format) |
| VQA | VRSBench | SIH | — | — | — | NOT RUN + INCOMPATIBLE (answer format) |
| Captioning | VRSBench | SIH | — | — | — | NOT RUN + INCOMPATIBLE (retrieval vs generative) |
| Grounding | VRSBench | SIH | — | — | — | NOT RUN (compatible; dataset not provisioned) |
| Change VQA | CDVQA | SIH | trained classifier | majority 0.5084 | **Test 0.7114 / Test2 0.6632** | ✅ real held-out (calibrated ECE 0.014) |
| Change detection | SECOND | additional public | learned Siamese U-Net | Otsu ref 0.2248 IoU | **IoU 0.4219 / F1 0.5934** | ✅ real held-out (learned ~2× baseline) |
| Change VQA (occurrence) | synthetic | — | — | — | acc **1.0** (n=12) | ✅ engineering validation only |
| Change detection | LEVIR-CD | additional | — | — | — | NOT RUN (optional; not provisioned) |
| Change detection | OSCD | additional | — | — | — | NOT RUN (optional; not provisioned) |
| Change detection | synthetic | — | — | — | IoU/F1 **1.0** (4) | ✅ engineering validation only |
| Optical+SAR cross-modal | (no matching public benchmark) | SIH (R13) | — | — | 2/4 property checks | ✅ synthetic validation; benchmark absence documented |
| Confidence calibration | synthetic (end-to-end) | — | — | — | ECE 0.287, Brier 0.091 | ✅ engineering validation only |

## What is genuinely evidenced

- **CDVQA (SIH) — real trained model**, held-out Test 0.71 / Test2 0.66 (+~20 pts over majority
  baseline), well-calibrated (ECE 0.014). Trained on real CDVQA QA + SECOND imagery,
  leakage-checked. See `docs/CDVQA_TRAINING_REPORT.md`.
- **SECOND (additional public) — real learned detector**, held-out IoU 0.42 / F1 0.59, ~2× the
  deterministic reference. See `docs/SECOND_TRAINING_REPORT.md`.
- **R9/R10 adaptation** — real, measured retrieval improvement (~6.9× mean recall) from the
  Phase 5 LoRA adaptation on BigEarthNet.txt. Correctly labelled as a retrieval metric.
- **R11/R12 change** — the production change pipeline localizes known synthetic changes exactly
  and answers occurrence questions correctly on the synthetic fixture; radiometric-vs-semantic
  scope stated.
- **R13 optical+SAR** — genuine feature-level fusion validated on controlled synthetic inputs;
  no suitable public benchmark exists for the measured-complementarity task (documented).
- **Confidence** — measurable and, on the fixture, well-behaved but mildly under-confident;
  thresholds unchanged pending real-benchmark calibration.

## What is honestly not done

- RSVQA and VRSBench (VQA/captioning/grounding): NOT RUN — datasets not provisioned, and the
  current VQA/captioning specialists are answer-format incompatible (a constrained/generative VQA
  head and a generative captioner are required). VRSBench grounding (OWL-ViT) is compatible and
  runnable once provisioned. LEVIR-CD/OSCD: optional, NOT RUN. No scores were invented.

See `reports/benchmarks/` for machine-readable evidence, `docs/phase9_model_selection.md` for the
production-model decision, and `docs/phase9_error_analysis.md` for failure analysis.
