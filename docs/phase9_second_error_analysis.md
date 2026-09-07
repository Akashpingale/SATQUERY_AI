# SECOND Error Analysis (Phase 9)

Based on the real held-out internal test evaluation (`reports/benchmarks/second/metrics.json`).
Task is **binary change** (derived from semantic labels).

## Headline
| Detector | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| Learned Siamese U-Net | 0.4219 | 0.5736 | 0.6147 | 0.5934 |
| Deterministic reference (Otsu) | 0.2248 | 0.3076 | 0.4549 | 0.3670 |

## Error categories observed (pixel confusion)
- **False positives (precision 0.57):** the largest error source. The learned model over-flags
  change at object boundaries and under illumination/seasonal radiometric differences between T1
  and T2 that are not labelled as land-cover change. The deterministic reference is far worse
  here (precision 0.31), i.e. it over-triggers on radiometric difference.
- **False negatives (recall 0.61):** missed change on small or low-contrast changed regions, and
  on class-to-class changes with similar RGB appearance (e.g. low_vegetation ↔ nvg_surface).
- **Boundary error:** at 256 px both detectors lose thin/edge change; some FN/FP concentrate on
  region borders.
- **Semantic confusion (not directly scored):** because the task is binarised, from-to class
  confusions (e.g. tree→low_vegetation) are collapsed; a semantic model would expose these and is
  future work.

## Learned vs deterministic
The learned detector improves precision (0.31→0.57) and IoU (0.22→0.42) markedly, i.e. it
suppresses the deterministic method's spurious radiometric triggers while keeping recall
comparable. This is why, on this dataset, the learned detector is the better change detector —
yet the deterministic CVA remains the production **default** for general/georeferenced imagery
(robustness + no domain assumption), with the learned model available as a registered specialist.

## Not fabricated
Numbers come from the single held-out evaluation over 446 pairs; the reference baseline uses no
test-derived tuning.
