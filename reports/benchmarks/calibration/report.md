# Confidence calibration (synthetic, end-to-end)

**Kind:** engineering-validation calibration (synthetic; small N; thresholds unchanged)

- Samples: 12  answered: 12
- ECE: 0.2868  MCE: 0.421  Brier: 0.0912
- Overall confidence: 0.7132  overall accuracy: 1.0
- Coverage: 1.0  selective accuracy: 1.0

Reliability bins (lower–upper: n, mean_conf, acc):
  - 0.0–0.2: n=0 conf=0.0 acc=0.0
  - 0.2–0.4: n=0 conf=0.0 acc=0.0
  - 0.4–0.6: n=2 conf=0.579 acc=1.0
  - 0.6–0.8: n=4 conf=0.6425 acc=1.0
  - 0.8–1.0: n=6 conf=0.805 acc=1.0

Interpretation: with a small synthetic N this measures whether the production confidence tracks correctness on this fixture; it is not a benchmark calibration and no thresholds were changed on the basis of it.
