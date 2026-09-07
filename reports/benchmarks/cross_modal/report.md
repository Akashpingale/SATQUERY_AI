# Cross-modal (optical+SAR) synthetic validation

**Kind:** engineering-validation (synthetic; not a real-world benchmark)

- Scenarios: 4
- Property checks passed: 2/4
- Fusion algorithm: xmodal-1.0.0

Per-check:
  - [FAIL] correlated: correlated pair less complementary than anticorrelated (corr.complementarity=0.0382 < anti=0.0375)
  - [FAIL] anticorrelated: anticorrelated pair is strongly complementary (>= 0.5) (complementarity=0.0375)
  - [PASS] sar_structure: SAR-only information/structure exceeds the flat-SAR baseline (sar_only_information=0.0, structured_fraction=0.04958767361111111 vs flat=0.0007595486111111111)
  - [PASS] water_agreement: cross-modal water agreement available with non-zero overlap (available=True, agree_fraction=0.14583333333333334, iou=0.5833333333333334)
