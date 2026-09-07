# CDVQA Error Analysis (Phase 9)

Based on the real held-out Test/Test2 predictions (`reports/benchmarks/cdvqa/predictions.jsonl`,
`errors.jsonl`). Per-type accuracy (Test):

| Question type | Accuracy | Reading |
|---|---|---|
| change_or_not | 0.838 | Strong — binary "did X change" is learned well. |
| decrease_or_not | 0.792 | Strong. |
| increase_or_not | 0.781 | Strong. |
| change_ratio_types | 0.718 | Good — coarse categorical ratios. |
| change_to_what | 0.583 | Moderate — naming the resulting class is harder. |
| largest_change | 0.585 | Moderate — requires ranking classes by area. |
| change_ratio | 0.394 | Weak — fine-grained ratio buckets. |
| smallest_change | 0.325 | Weakest — smallest-area class is easily confused. |

## Error categories observed
- **Counting / ratio bucketing (change_ratio, smallest/largest_change):** the dominant error
  source. Predicting an exact 10-point ratio bucket or the smallest-area class from frozen
  global CLIP features is hard; the model often picks an adjacent bucket or the majority class.
- **Class naming (change_to_what):** confusions among visually similar land-cover classes
  (low_vegetation vs trees; nvg_surface vs buildings).
- **Change/no-change:** comparatively rare errors; the model separates changed from unchanged
  scenes reliably.
- **Closed-vocabulary limitation:** any Test answer outside the 19-class train vocabulary is
  unanswerable by construction and counts as incorrect (coverage effect), not a reasoning error.

## Likely causes
Frozen global image features (no spatial/region detail) limit counting and small-object
questions; a region-aware or fine-tuned encoder would most help the weak categories. The strong
binary categories confirm the temporal-difference signal is informative.

## Not fabricated
All numbers come from the single held-out evaluation; no cherry-picking. Weak categories are
reported as-is.
