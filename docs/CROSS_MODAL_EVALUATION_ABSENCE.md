# Optical + SAR Cross-Modal Evaluation — Documented Benchmark Absence

**Status: no suitable public benchmark was used, because none directly matches the implemented
task.** Rather than force an incompatible dataset into a fabricated score, this file records what
was searched, why it does not fit, what the system actually measures, how it was validated
synthetically, and what a real benchmark would require.

## What the system implements (R13)
Genuine **feature-level** optical+SAR fusion (`ml/inference/fusion.py`, `analyze_cross_modal`):
co-register optical+SAR onto one grid, extract per-modality features, then measure
cross-modal quantities neither image produces alone — Pearson correlation of optical brightness
vs SAR backscatter, complementarity (1−|corr|), an optical-only / SAR-only / shared information
partition, and optical-NDWI vs SAR-dark **water agreement**. It is not caption concatenation.

## Datasets / tasks considered and why they do not directly fit
- **RSVQA / VRSBench / CDVQA:** optical (or temporal-optical) VQA/caption/grounding — no SAR, no
  cross-modal complementarity target. Not applicable.
- **SEN1-2 / SEN12MS / BigEarthNet-MM (optical+SAR):** paired optical+SAR tiles, but the labels
  are land-cover **classification** or the task is **translation/retrieval**, not "measured
  complementarity". They could support a *classification* benchmark of a different component, not
  this measurement.
- **Change-detection sets (LEVIR-CD/OSCD/SECOND):** single-modality change, not optical+SAR
  fusion.
- There is **no widely-adopted public benchmark** whose ground truth is "how complementary are
  these optical and SAR views", which is what this component quantifies. Complementarity here is a
  measurement, not a labelled prediction, so standard accuracy/IoU do not apply.

## Synthetic validation performed (engineering only)
`ml/evaluation/cross_modal.py` builds controlled optical+SAR fixtures and runs the production
fusion, checking that measurable quantities respond to controlled inputs. Executed results
(`reports/benchmarks/cross_modal/`): a SAR-only structured block raises SAR structure and
SAR-only information above the flat-SAR baseline (**pass**); a shared optical-NDWI + SAR-dark
strip yields available water agreement with IoU ≈ 0.58 (**pass**); two ramp-based
complementarity checks did **not** pass (honest — see `docs/phase9_error_analysis.md`: under a
smooth ramp |correlation| is high for both correlated and anticorrelated fixtures, so
complementarity is low in both — a fixture-probe limitation, not an algorithm bug).

**This is synthetic engineering validation, never real-world benchmark accuracy.**

## What a real benchmark would require
Either (a) a labelled optical+SAR dataset with a defined complementarity/agreement ground truth
(none known), or (b) redefining the evaluation around a *downstream* task with labels — e.g.
optical+SAR land-cover or flood mapping — which would evaluate a **different** (classification)
component, not the measured-complementarity analysis shipped today. Building such a component and
provisioning SEN12MS-style data is scoped as future work; it is not claimed here.
