# Spec 14 — Evaluation (stub)

> Phase 9. Stub now; full `design.md` + `tasks.md` authored when the phase begins.

## Introduction
Real, reproducible evaluation of the implemented capabilities. Never fabricate numbers;
report `NOT EVALUATED — dependency unavailable` where applicable.

## Requirements (high level)
- Metrics where applicable: VQA accuracy, caption quality, grounding IoU/localization,
  change detection/answer accuracy, fusion quality, routing accuracy, confidence
  calibration, latency, memory, failure rate.
- Reproducible harness with fixed seeds and documented datasets/splits.

## Design (outline)
`ml/evaluation/` harness + `configs/`; outputs to `reports/` and `docs/evaluation.md`.

## Implementation tasks (outline)
Define metrics per task → datasets/splits → harness → runs → report generation →
calibration analysis.

## Dependencies
Specialists (08–11), agent (05), datasets (07/03). Possibly cloud GPU.

## Acceptance criteria
- Metrics computed on real data with reproducible configs; gaps marked NOT EVALUATED.

## Tests
Metric-function unit tests; harness smoke test on tiny subsets.

## User inputs required
- Access to any private/benchmark evaluation datasets (ask when reached).

## Known risks
- Compute limits; dataset access/licensing; honest reporting of partial coverage.
