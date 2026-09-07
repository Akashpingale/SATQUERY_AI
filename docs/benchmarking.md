# Benchmarking & Evaluation (Phase 9)

This document explains how SatQuery AI is evaluated: which benchmarks are supported, how to
acquire the datasets, the commands, the metrics, model versions, baseline-vs-adapted
comparison, confidence calibration, limitations, and reproducibility. A new developer should be
able to reproduce every executed number by following it.

**Guiding principle:** Phase 9 reports *evidence*, not appearance. No number here is fabricated.
Where a benchmark could not be run it is marked `NOT RUN — <reason>`, never `0%`.

---

## 1. What was actually executed

Everything in this section runs **offline** and routes through the **production** specialists.

| Evaluation | Kind | Command |
|---|---|---|
| Baseline vs adapted **retrieval** (BigEarthNet.txt) | Real public dataset, held-out test | `python scripts/evaluate_phase5.py --split test` |
| Change detection (synthetic known masks) | Engineering validation | `python scripts/run_benchmarks.py --suite change_detection` |
| Change-occurrence VQA (synthetic) | Engineering validation | `python scripts/run_benchmarks.py --suite change_vqa` |
| Cross-modal optical+SAR (synthetic) | Engineering validation | `python scripts/run_benchmarks.py --suite cross_modal` |
| Confidence calibration (synthetic, end-to-end) | Engineering validation | `python scripts/run_benchmarks.py --suite calibration` |

Outputs: `reports/benchmarks/<suite>/{metrics.json,predictions.jsonl,config.json,report.md}` and
`reports/benchmarks/model_comparison.json`.

### Metric-family separation (critical)

- **Retrieval recall** (BigEarthNet.txt) is image↔text ranking. It is **not** VQA, captioning,
  or grounding accuracy and is never combined with them.
- **Change detection** uses binary segmentation metrics (IoU/precision/recall/F1) against a
  reference mask, plus change/no-change occurrence accuracy.
- **Change VQA** uses classification accuracy with an explicit answered/abstained split.
- **Calibration** uses ECE/MCE/Brier/reliability/selective-accuracy over `(confidence, correct)`.

---

## 2. Dataset inventory & classification

### SIH26167-prescribed datasets

| Dataset | Role | Provisioned here | Source |
|---|---|---|---|
| **BigEarthNet.txt** | RS VLM adaptation (Phase 5) + retrieval eval | ✅ yes | BigEarthNet v2 imagery + BigEarthNet.txt captions |
| **RSVQA** | single-image VQA | ❌ no | https://rsvqa.sylvainlobry.com/ |
| **VRSBench** | VQA / captioning / grounding | ❌ no | https://vrsbench.github.io/ |
| **CDVQA** | multitemporal change VQA | ❌ no | https://github.com/YZHJessica/CDVQA *(user-provided source)* |

### Additional public datasets (NOT SIH-prescribed)

| Dataset | Role | Provisioned here | Source |
|---|---|---|---|
| **SECOND** | semantic change detection (training/external validation) | ❌ no | User-provided Google Drive link (`SECOND_train_set.rar`) |
| LEVIR-CD | binary change detection | ❌ no | https://chenhao.in/LEVIR/ |
| OSCD | binary change detection | ❌ no | https://rcdaudt.github.io/oscd/ |

> **SECOND is an *additional public* dataset used to improve and/or externally validate the
> change-detection component. It is NOT SIH-provided, NOT ISRO-provided, and NOT an official SIH
> benchmark.** The two user-provided sources (CDVQA GitHub, SECOND Google Drive) are recorded in
> `ml/evaluation/datasets.py` and above.

**Never used for training or evaluation:** hidden ISRO/SAC evaluation imagery/annotations.

Availability is checked (without downloading) by:

```
python scripts/eval_external.py --all
```

Datasets are **not** downloaded by tests, Docker builds, or app startup. Provisioning is a
manual, licence-aware step: place the prepared dataset at the `expected_path` recorded in
`ml/evaluation/datasets.py` (e.g. `data/benchmarks/cdvqa`), then re-run the command.

---

## 3. Baseline vs adapted VLM (retrieval)

Executed in Phase 5 and reproducible with `scripts/evaluate_phase5.py`. Held-out BigEarthNet.txt
test split (2053 samples, identical preprocessing for both models):

| Metric | Baseline (`clip-rsicd-v2`) | Adapted (`+LoRA ben-txt r16a32`) |
|---|---|---|
| mean recall | 0.0115 | **0.0791** (~6.9×) |
| i2t R@1 | 0.0054 | 0.0346 |
| t2i R@1 | 0.0015 | 0.0185 |

This is the genuine, measured benefit of the Phase 5 adaptation — as a **retrieval** metric.
It evidences SIH R9/R10 (real remote-sensing VLM adaptation). It does **not** establish VQA or
caption accuracy.

---

## 4. Single-image VQA / captioning / grounding (RSVQA, VRSBench)

**Status: NOT RUN — datasets not provisioned; and a documented architecture incompatibility.**

- **VQA (RSVQA, VRSBench-VQA):** the current VQA specialist is zero-shot CLIP **concept scoring**
  over a fixed vocabulary. RSVQA/VRSBench answers are yes/no, counts, and area/comparison
  classes. The specialist cannot emit those answer types, so a direct accuracy would be
  near-zero and misrepresentative. A valid run requires a constrained/generative VQA head — a
  new trainable component (parameter-efficient head or LoRA on a VQA-capable backbone). The
  adapter + normalization scaffolding is in place (`ml/evaluation/metrics.py`,
  `ml/evaluation/datasets.py`); training/eval is future work.
- **Captioning (VRSBench):** the captioner is **retrieval-composition**, not generative, so it
  cannot produce valid predictions for generative-caption metrics (BLEU/CIDEr/etc.). We do not
  report a caption score for it. A generative captioner would be a new component.
- **Grounding (VRSBench):** the OWL-ViT grounding specialist is **zero-shot** and box-IoU
  compatible. It is genuinely evaluable with VRSBench grounding annotations (Recall@IoU / IoU),
  but VRSBench is not provisioned here, so it is `NOT RUN`. Model confidence is never reported as
  grounding accuracy.

---

## 5. Change VQA (CDVQA)

**Real CDVQA: NOT RUN — dataset not provisioned; published field schema unverified.**

The measurement-grounded change-VQA specialist answers occurrence / location / extent /
NDVI-NDWI-direction questions and **declines** semantic-class questions (e.g. building type) by
design. A documented JSONL adapter (`ml.evaluation.change_vqa.load_samples` /
`predict_all` / `score_predictions`) converts a real CDVQA release and routes it through the
production temporal pipeline. Coverage is always reported next to accuracy.

**Synthetic proxy (executed):** a change-**occurrence** fixture with independent yes/no ground
truth gives accuracy **1.0** (12 samples, Wilson 0.76–1.0). This is engineering validation of
occurrence detection, explicitly **not** CDVQA accuracy, and it covers only occurrence — not
semantic change classification.

> **Measured vs semantic change.** The detector measures *radiometric/pixel/spatial* change. It
> does not classify *semantic* change (construction, demolition, vegetation loss, etc.). CDVQA
> semantic questions are outside the current capability and are declined, not guessed.

---

## 6. Change detection (SECOND / LEVIR-CD / OSCD)

**Real datasets: NOT RUN — not provisioned.** The detector `cva-robust` is deterministic binary
radiometric change detection. On real datasets it is expected to flag broad radiometric
differences rather than only the semantic class of interest (buildings for LEVIR-CD; from-to
classes for SECOND); scores must be read with that caveat, and any semantic→binary label
collapse must be documented (see `ml/evaluation/datasets.py` SECOND note).

**Synthetic proxy (executed):** production `detect_change` on known-mask fixtures gives micro
IoU / F1 **1.0** and occurrence accuracy **1.0** over 4 scenarios (including a no-change
specificity case). These are clean synthetic blocks and demonstrate correctness of the pipeline
and metric wiring, not real-world difficulty.

The Phase 6 deterministic detector is **retained**; any future learned detector (e.g. trained on
SECOND) would be an additional registry specialist selected by the agent, never a silent
replacement.

---

## 7. Optical + SAR cross-modal

No public benchmark matches this exact task (measured cross-modal complementarity, not a
labelled classification/segmentation target). See `docs/CROSS_MODAL_EVALUATION_ABSENCE.md`.
Synthetic property validation is executed (`reports/benchmarks/cross_modal/`): SAR-only structure
and water-agreement checks pass; the two ramp-based complementarity checks did not pass under the
chosen fixtures and are reported honestly rather than tuned.

---

## 8. Confidence calibration

Measured end-to-end through the orchestrator on the synthetic change-occurrence set:

- ECE **0.287**, MCE 0.421, Brier **0.091**; overall accuracy 1.0 at mean confidence 0.71.
- Interpretation: on this easy fixture the system is **never wrong but mildly under-confident**
  (confidence sits ~0.58–0.81 while accuracy is 1.0). Small N (12); the reliability diagram is
  coarse.
- **No thresholds were changed** on the basis of this. Re-tuning `HIGH_THRESHOLD` /
  `MEDIUM_THRESHOLD` requires calibration on a real benchmark **validation** split, evaluated on
  a held-out **test** split after freezing — which is future work pending dataset provisioning.
- Calibration must never be fitted on test answers (leakage). The framework
  (`ml/evaluation/calibration.py`) supports reliability/ECE/Brier/selective-accuracy/abstention;
  temperature/Platt/isotonic scaling would be fit on validation data when a real benchmark is
  provisioned.

---

## 9. Reproducibility

Every run writes a `config.json` capturing: dataset + split, seed, Python/NumPy/Torch/
Transformers/PEFT versions, device (and GPU/VRAM when CUDA), git commit (`HEAD / uncommitted`
when the repo has no commit — never fabricated), timestamp, and model/tool versions. See
`ml/evaluation/reproducibility.py`.

Environment used for the executed runs: Python 3.11, CPU (the synthetic algorithmic suites need
no GPU); the Phase 5 retrieval numbers were produced on an RTX 2050 (4 GB) — see
`reports/phase5/environment.json`.

---

## 10. Limitations (honest)

- The single-image VQA/captioning benchmarks are architecture-incompatible with the current
  zero-shot/retrieval specialists; real RSVQA/VRSBench-VQA/caption numbers require new trained
  components. Not fabricated here.
- CDVQA / SECOND / LEVIR-CD / OSCD are not provisioned in this environment, so their real numbers
  are `NOT RUN`. Adapters, availability checks, and reproducible commands are provided.
- Synthetic suites validate engineering correctness, not real-world difficulty; their high
  scores must not be read as real-world accuracy.
- Calibration N is small; it characterises behaviour on a fixture, not a benchmark.


---

## 11. Real dataset training (CDVQA + SECOND) — provisioned & executed

The user provisioned CDVQA and SECOND locally; both were trained and evaluated on held-out data.
This supersedes §4–§6's "NOT RUN" status **for CDVQA and SECOND**.

- **CDVQA (SIH):** constrained classifier (frozen CLIP `clip-rsicd-v2` features + GRU question
  encoder + fusion). Held-out **Test 0.7114 / Test2 0.6632** (majority baseline 0.5084/0.4560),
  leakage 0, calibrated (Val→Test temperature scaling, ECE 0.036→0.014). Commands:
  `python scripts/train_cd_vqa.py --epochs 20 --batch-size 256 --seed 42` then
  `python scripts/calibrate_cdvqa.py`. Report: `docs/CDVQA_TRAINING_REPORT.md`.
- **SECOND (additional public, NOT SIH):** learned Siamese U-Net for binary change (derived from
  semantic labels; documented). Held-out **IoU 0.4219 / F1 0.5934** vs deterministic reference
  0.2248 / 0.3670. Command:
  `python scripts/train_second_change.py --epochs 12 --batch-size 8 --size 256 --seed 42`.
  Report: `docs/SECOND_TRAINING_REPORT.md`.
- Both integrated as registered specialists (`cdvqa_trained`, `second_change_learned`), kept
  `degraded` and non-default (deterministic detectors remain default). Leakage protection:
  `docs/phase9_data_leakage.md`. Error analysis: `docs/phase9_cd_vqa_error_analysis.md`,
  `docs/phase9_second_error_analysis.md`.
- **Still NOT RUN:** RSVQA, VRSBench, LEVIR-CD, OSCD (not provisioned; RSVQA/VRSBench VQA/caption
  architecture-incompatible).
