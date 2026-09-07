# Spec 10 — Temporal Change

> Phase 6. **Implemented.** Requirements below are the acceptance contract; `design.md` records
> what was built and `tasks.md` tracks completion. Method and measured behaviour are in
> `docs/phase6.md`, `docs/evaluation.md`, and `docs/limitations.md`.

## Introduction
Bi-temporal (T1/T2) analysis of the same area: validate compatibility, co-register, measure
change, and produce a change description and change-VQA with visual evidence and evidence-based
confidence. Optional change map. A generic "describe the difference" is not enough — the answer
must be grounded in a real per-pixel measurement.

## Requirements

### R10.1 — Validate the temporal pair before analysing
Geographic compatibility, CRS, extent, resolution, and alignment must be checked first, and an
incompatible pair rejected with a clear reason rather than analysed anyway.

**Met.** The `registration_validator` step runs first in the toolchain and short-circuits the
workflow when the pair is incompatible (`check_pair_compatibility`). Non-overlapping,
insufficient-overlap, and (unreferenced) dimension-mismatch pairs are rejected with
`incompatible_inputs`. Verified by `test_temporal_api.py`.

### R10.2 — Co-register safely, and record what was done
Where alignment/resampling is required it must be performed explicitly and recorded in the
execution trace (geospatial rule).

**Met.** `geospatial.align_pair` reads both dates onto one grid via `WarpedVRT`: reprojects a
CRS mismatch, resamples to the **coarser** resolution so nothing is upsampled, crops to the
overlap, and caps memory by decimation. Every operation is recorded and surfaced as a
`Co-registration` trace step. A pair that cannot be aligned raises `AlignmentError` and is not
compared.

### R10.3 — Real change detection, not raw RGB differencing
Change must be measured with a defensible remote-sensing method and report its own uncertainty.

**Met.** `ml.inference.detect_change`: radiometric normalisation (robust, two-pass) → change
vector analysis (SAR log-ratio) → robust noise-floor threshold → binary opening → connected
components. Registration quality, coherence, and changed fraction are all measured and reported.
Deterministic; version `cva-robust-1.1.0`. Evaluated against synthetic known-truth in
`test_change_detection.py`.

### R10.4 — Change description AND change-VQA
Deliver at least one; both if feasible.

**Met — both.** `TemporalChangeSpecialist` (grounded description) and `ChangeVQASpecialist`
(occurrence / location / magnitude / vegetation / water / built-up / which-date), sharing one
measurement. Wording states only measured quantities; unanswerable questions return a limitation.

### R10.5 — Visual evidence and optional change map
Change overlay; change map with area/percentage only when geospatially valid.

**Met.** `change_map.py` renders a magnitude-ramped change map and a T1/T2/change strip from the
measured mask, served via the evidence-token endpoint. Per-region geometry carries real
coordinates. Area is reported in m² only for metric CRSs, else percentage/pixels.

### R10.6 — Evidence-based confidence, auditable trace, no fabrication
Confidence from measurable signals; trace observable with no chain-of-thought; no invented
change, location, area, or cause.

**Met.** `measurement_score` derives strength from registration, coherence, changed-fraction
margin, and magnitude headroom; unanswerable → "Unable to determine". Trace records
co-registration, registration check, and the change measurement (threshold included). Cause is
never asserted; built-up questions refuse the classification; misregistration and cross-modality
pairs are refused.

### R10.7 — Benchmark readiness without benchmark claims
Prepare for CDVQA-style evaluation without claiming results.

**Met.** `ml/evaluation/change_vqa.py` provides a documented sample schema, a predictor driving
the real pipeline, and coverage-aware metrics. The CDVQA schema is documented as **unverified**
(dataset not obtained). No CDVQA result is claimed anywhere. Running it is Phase 9.

## Dependencies
Raster ingestion (03), registry (06), agent (05), evidence/confidence (12). Libraries: NumPy,
SciPy (`ndimage`), Rasterio (`WarpedVRT`), pyproj. No model weights, no GPU.

## Acceptance criteria
- [x] Compatible pairs yield a measured change answer + evidence; incompatible pairs rejected.
- [x] Co-registration performed and recorded in the trace.
- [x] Both change description and change-VQA implemented.
- [x] Change map rendered from the measured mask; area only when geospatially valid.
- [x] Confidence evidence-based; trace has no chain-of-thought.
- [x] No fabricated change, location, area, or cause; honest limitations returned.
- [x] Phase 1–5 regression intact; ruff/eslint/tsc/build/compose green.

## Tests
`test_change_detection.py` (unit: alignment, band roles, detector against known truth, rendering)
and `test_temporal_api.py` (integration: registry, planner, both specialists, negatives,
trace/evidence, evaluation adapter). Change detection runs with no weights, so it is always
exercised in CI; the optional land-cover reading is gated on `models_ready`.

## Resolved risks
- ~~Registration errors propagate to change quality.~~ Measured and gated: >1 px misregistration
  → inconclusive; detection vs interpretation kept strictly separate.

## Outstanding (later phases)
- Benchmark change-detection quality and CDVQA accuracy; calibrate confidence — Phase 9.
- A learned change model behind the same interface; multi-temporal (>2 date) analysis; semantic
  change-type classification — not scheduled.
