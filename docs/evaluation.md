# Evaluation

What has actually been measured, and — just as importantly — what has not. Nothing in this
file is estimated, extrapolated, or copied from a paper. Anything unmeasured says so.

| Capability | Evaluated? | Where |
|---|---|---|
| Phase 5 adaptation (baseline vs adapted) | **Yes** — held-out retrieval | This page |
| Phase 4 VQA accuracy | **NOT EVALUATED** — no labelled benchmark set downloaded | Phase 9 |
| Phase 4 captioning quality | **NOT EVALUATED** — no reference captions for our test rasters | Phase 9 |
| Phase 4 grounding accuracy | **NOT EVALUATED** — no bounding-box ground truth used | Phase 9 |
| Phase 6 change detection — correctness | **Yes** — measured on synthetic known-truth pairs | This page |
| Phase 6 change detection — accuracy | **NOT BENCHMARKED** — no labelled CD dataset run | Phase 9 |
| Change VQA accuracy (CDVQA) | **NOT BENCHMARKED** — dataset not downloaded; interface ready | Phase 9 |
| Phase 7 optical–SAR fusion — correctness | **Yes** — measured on synthetic known-truth pairs | This page |
| Phase 7 optical–SAR fusion — accuracy | **NOT BENCHMARKED** — no suitable public benchmark run | Phase 9 |
| Standard benchmarks (RSVQA, VRSBench, LEVIR-CC, CDVQA) | **NOT EVALUATED** — datasets not downloaded | Phase 9 |
| Latency | Yes (CPU, measured) | `docs/models.md`, `docs/phase7.md` |

---

## Phase 5 — did adaptation actually work?

**Task.** Image↔text retrieval on the BigEarthNet.txt Lithuania/Summer captioning subset.

**Why retrieval.** The adapted model is a contrastive dual encoder, and retrieval is the
objective-aligned way to measure it: given an image, how highly does the model rank that
image's true caption among all 2,053 candidates, and vice versa. It needs no extra labels and
no generative decoding, so there is nothing to fabricate.

### Protocol

| | |
|---|---|
| Split | `test` — **2,053 samples** from the dataset's own `split` column |
| Leakage control | Never used for training, validation, model selection, or hyperparameters |
| Candidate pool | All 2,053 captions (i2t) / all 2,053 images (t2i) |
| Preprocessing | **Identical** for both models |
| Device | CUDA |
| Metrics | Recall@1/5/10 and median rank, both directions |
| Encode time | baseline 36.0 s, adapted 36.8 s |

Both models were evaluated in the same process, in the same run, on the same data, with the
same preprocessing. The only difference is whether the LoRA adapter is attached.

**Chance level for this pool size** (the number any result must be read against):
R@1 = 0.00049, R@5 = 0.00244, R@10 = 0.00487, median rank 1026.5.

### Results (measured)

| Metric | Baseline | Adapted | Change |
|---|---|---|---|
| image→text R@1 | 0.0054 | **0.0346** | +0.0292 |
| image→text R@5 | 0.0166 | **0.0867** | +0.0701 |
| image→text R@10 | 0.0253 | **0.1432** | +0.1179 |
| text→image R@1 | 0.0015 | **0.0185** | +0.0170 |
| text→image R@5 | 0.0058 | **0.0697** | +0.0638 |
| text→image R@10 | 0.0146 | **0.1218** | +0.1072 |
| **mean recall** | **0.0115** | **0.0791** | **+0.0675** |
| image→text median rank | 518.0 | **85.0** | −433 |
| text→image median rank | 645.0 | **84.0** | −561 |

Source: `reports/phase5/evaluation_metrics.json`.

### What this does and does not establish

**It establishes that adaptation worked.** Mean recall improved **6.9×** (0.0115 → 0.0791),
every one of the six recall metrics improved, and both directions improved together — which
is what genuine dual-encoder adaptation looks like, as opposed to one tower degenerating.
The median-rank shift is the clearest signal: the correct caption moved from rank 518 to
rank **85** out of 2,053, and the correct image from 645 to **84**. Both models sit far above
the 0.00049 chance floor, so the baseline was already better than random and the adapter
improved on a real starting point.

**It does not establish accuracy.** Read the absolute numbers honestly: adapted R@1 of 0.0346
means the exact right caption is ranked first **3.5 % of the time**. Three things explain
that, and none of them are hidden:

1. **The task is deliberately hard.** 2,053 near-identical candidates, all Lithuanian summer
   land cover, all opening with the same boilerplate sentence. Distinguishing "mixed forest
   with some arable land" from "coniferous forest with some arable land" as an exact top-1
   match is much harder than answering a user's question about which land cover dominates.
2. **84 % of captions are truncated at CLIP's 77-token limit**, and on average only 49 % of a
   caption's tokens reach the model (measured — see `docs/datasets.md`). A large share of the
   text that would distinguish two candidates is never seen.
3. **4,008 training samples** is small. Validation loss was still falling at epoch 6, so this
   is a data ceiling, not a converged result.

**It does not transfer as a general claim.** This measures one country, one season, one
sensor, one resolution. It says nothing about Indian land cover, arid or tropical regions, SAR
imagery, or sub-metre resolution. The hidden ISRO/SAC set will differ on most of those axes.

**It does not measure the user-facing tasks.** Retrieval improving does not automatically mean
VQA answers or captions improved by the same factor. It means the shared embedding space that
those specialists read from is measurably better aligned with remote-sensing language. That is
a real and relevant improvement, and it is a different claim from "VQA accuracy improved" —
which has **not** been measured and is not asserted anywhere in this repository.

### Why the registry still says `degraded`

Both adapted entries (`vqa_adapted`, `captioning_adapted`) are registered as `degraded`, not
`available`, and stay that way after these results. Reasons: scores remain **uncalibrated**
(a 0.70 similarity is not a 70 % probability of correctness); the improvement is measured on
**one geographic/seasonal subset**; and no user-facing task accuracy has been benchmarked.
`available` requires established accuracy on the capability itself, which is Phase 9 work.

### Verified end to end, not just offline

`scripts/verify_phase5.py` runs **25 checks** in a fresh process and all 25 pass, including
the ones that matter for honesty:

- the adapter loads from disk and reports its real rank/alpha/targets (16 / 32 /
  `k_proj,out_proj,q_proj,v_proj`);
- adapted output **measurably differs** from baseline on the same input (max |Δprobability|
  0.115805) — proving the adapter is not a no-op;
- the planner selects `vqa_adapted` / `captioning_adapted` and records why;
- a real upload → analysis request through the API returns HTTP 201, and its **trace** names
  the base model, adapter path, and training dataset;
- confidence stays honest (Medium, 0.551) rather than being inflated by adaptation;
- **grounding still routes to the unchanged Phase 4 tool** — no unearned improvement is
  claimed for it;
- counting questions are still refused.

### Reproduce

```powershell
$env:HF_HOME="D:\SatQuery-AI\models\hf-cache"
.\.venv\Scripts\python.exe scripts\evaluate_phase5.py --split test
.\.venv\Scripts\python.exe scripts\verify_phase5.py
```

`--split validation` re-runs the same protocol on the validation split. Note that validation
*was* used during training, so only `test` supports a held-out claim.

---

## Phase 6 — is the change detection correct?

**Measured (correctness, not benchmark accuracy).** `backend/tests/test_change_detection.py`
asserts the detector against known synthetic ground truth, because a synthetic pair is the only
case where the true changed area is known exactly:

| Input | Expected | Measured |
|---|---|---|
| Identical pair | 0 changed pixels | 0 |
| Illumination-only difference (×1.25 + 300) | 0 (normalised away) | 0 |
| Independently re-noised pair | 0 coherent regions | 0 |
| A 30×40 block set to a new value | 1200 changed pixels, 1 region | exactly 1200, 1 region |
| Two separate blocks | 2 regions, correct sizes | 2, correct |
| Dominant + weaker change together | both survive thresholding | both detected |
| Whole scene shifted 2 px | inconclusive (misregistered) | inconclusive |
| NIR collapse with declared bands | NDVI falls, larger in changed area | measured, sign correct |

These prove the algorithm measures what it should and refuses what it cannot. They are **not**
a real-world accuracy claim — the imagery is synthetic.

**Not benchmarked.** No labelled change-detection dataset (LEVIR-CD, OSCD) and no CDVQA dataset
have been downloaded or run. `ml/evaluation/change_vqa.py` is a benchmark-ready interface: it
drives the real pipeline and reports **coverage alongside accuracy** (and accuracy over the
answered subset separately), so a system that declines most questions cannot look strong. The
CDVQA field/answer schema is documented as **unverified** because the dataset was not obtained;
converting a real release into the documented JSONL is an explicit Phase 9 adapter step. No
CDVQA number is reported anywhere in this repository.

## Latency

Measured CPU latencies for the serving path are in `docs/models.md` (model load, concept
scoring cold/warm, detection). They come from `scripts/check_models.py`. GPU serving latency
has **not** been measured — local serving is CPU by default.

## Test suite

The test suite verifies behaviour, not model accuracy. It never asserts against fabricated
model outputs: model-dependent tests skip when weights are absent (`models_ready`,
`adapter_present`, `training_reports` fixtures) and CI runs fully offline
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) so nothing downloads mid-test.

Current counts and how to run them are in `README.md`.
