# Change detection synthetic validation

**Kind:** engineering-validation (synthetic known masks; not LEVIR-CD/OSCD/SECOND)

- Detector: cva-robust (deterministic radiometric change vector analysis)
- Scenarios: 4
- micro IoU: 1.0  precision: 1.0  recall: 1.0  F1: 1.0
- mean IoU (changed pairs): 1.0
- occurrence accuracy: 1.0
- pixels: {'tp': 4044, 'fp': 0, 'fn': 0, 'tn': 32820}

Detects radiometric change, not semantic class. no-change pairs verify specificity; block pairs verify localization against a known mask.
