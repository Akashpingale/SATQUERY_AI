# SatQuery AI — Benchmark Evidence (Phase 9)

This directory holds the **machine-readable evidence** for every evaluation that was actually
executed, plus honest `NOT RUN` records for benchmarks whose datasets are not provisioned.

**Integrity rules honoured here:** no fabricated scores; metric families kept separate
(retrieval recall is never reported as VQA/caption/grounding accuracy); synthetic
engineering-validation is never labelled real-world benchmark accuracy; no hidden ISRO/SAC
data; test splits never used for training or calibration.

## Executed (real, this environment, offline)

| Evidence | What it is | Key result | Files |
|---|---|---|---|
| Baseline vs adapted **retrieval** | BigEarthNet.txt held-out test (2053), image-text retrieval | mean recall 0.0115 → **0.0791** (~6.9×) | `../phase5/evaluation_metrics.json`, `model_comparison.json` |
| Change detection (synthetic) | Production `cva-robust` vs known masks | IoU/F1 **1.0**, occurrence acc **1.0** (4 scenarios) | `change_detection/` |
| Change-occurrence VQA (synthetic) | Production temporal pipeline, yes/no ground truth | accuracy **1.0** (12; Wilson 0.76–1.0) | `change_vqa/` |
| Cross-modal fusion (synthetic) | Production `analyze_cross_modal` property checks | 2/4 directional checks pass (see note) | `cross_modal/` |
| Confidence calibration (synthetic) | End-to-end via orchestrator; ECE/Brier/selective | ECE **0.287**, Brier **0.091**, acc 1.0, conf 0.71 (under-confident) | `calibration/` |

The synthetic suites are **engineering validation** of the shipped algorithms, not real-world
benchmark accuracy. The cross-modal check "2/4" is an honest result: the two directional
complementarity checks did not pass under the specific ramp fixtures (the algorithm's
complementarity for smoothly-correlated *and* anticorrelated ramps is similarly low because
|correlation| is high in both); the SAR-structure and water-agreement checks pass. This is
reported as-is rather than tuned to look green — see `cross_modal/report.md`.

## NOT RUN (datasets not provisioned; no fabricated numbers)

RSVQA, VRSBench, CDVQA, SECOND, LEVIR-CD, OSCD — see `external/*.json` and
`docs/benchmarking.md` for the reason, the architectural compatibility caveats (e.g. the
zero-shot CLIP concept-scoring VQA cannot emit RSVQA yes/no/count answers), and the exact
reproducible command per dataset.

## Reproduce

```
python scripts/run_benchmarks.py --suite all        # synthetic suites + calibration
python scripts/evaluate_phase5.py --split test       # baseline vs adapted retrieval
python scripts/eval_external.py --all                # external dataset availability/status
```
