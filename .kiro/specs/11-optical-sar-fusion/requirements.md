# Spec 11 — Optical-SAR Fusion

> Phase 7. **Implemented.** Requirements below are the acceptance contract; `design.md` records
> what was built and `tasks.md` tracks completion. Method and measured behaviour:
> `docs/phase7.md`, `docs/evaluation.md`, `docs/limitations.md`.

## Introduction
Genuine joint analysis of co-registered optical + SAR imagery, demonstrating complementary
information (optical = spectral/contextual, SAR = structural/backscatter) via real multimodal
integration — **not** text concatenation of two separate model outputs.

## Requirements

### R11.1 — Validate the cross-modal pair
Exactly one optical and one SAR image of the same area; format/readability/CRS/overlap checked;
refuse two-optical, two-SAR, missing-modality, and non-overlapping pairs with a clear reason.

**Met.** `fusion_support.resolve_modalities` enforces one-of-each by recorded modality; the
`registration_validator` step rejects incompatible pairs before fusion. Verified in
`test_fusion_api.py`.

### R11.2 — Co-register to a common grid, recorded in the trace
Reuse Phase 6 alignment; bring both modalities to one grid; record reprojection/resample/crop.

**Met.** `geospatial.align_modalities` reuses the extracted Phase 6 grid planner, keeps all
optical bands and the SAR band (no truncation), resamples to the coarser resolution, crops to
the overlap, caps at 768 px, and records every operation as a `Co-registration` trace step.

### R11.3 — SAR preprocessing without invented units
Log-domain representation where justified; robust dynamic range; no calibrated units unless the
metadata supports them.

**Met.** SAR is averaged in linear domain, log10-scaled, robust-normalised, and reported as
**uncalibrated backscatter** unless metadata declares calibration. sigma0/gamma0 is never
assumed; SAR evidence is measured backscatter/roughness, never land-cover classes.

### R11.4 — Optical preprocessing from declared bands
Reuse the raster stack; NDVI/NDWI only when the required bands are declared.

**Met.** Brightness, spectral variance, and texture always; NDVI/NDWI only via
`resolve_band_roles`. Band order is never guessed.

### R11.5 — Real feature-level cross-modal fusion + complementary evidence
Implement genuine integration; distinguish optical, SAR, and cross-modal complementary evidence.

**Met.** `analyze_cross_modal` builds a joint feature representation and measures
correlation/complementarity, the optical-only / SAR-only / shared information partition,
optical↔SAR water agreement (IoU), and SAR-only structure regions — quantities no single
modality produces. Not caption concatenation.

### R11.6 — Cross-modal VQA
Answer natural-language cross-modal questions from measured evidence; honest limitation
otherwise.

**Met.** `OpticalSARFusionSpecialist` handles describe / complementary / what-SAR-reveals /
which-modality / agreement, grounded in the measurement; unsupported questions return a stated
limitation with "Unable to determine".

### R11.7 — Agent routing, registry, evidence, confidence, trace
Automatic routing (no manual fusion mode); registered tool; evidence-token artifacts;
evidence-derived confidence; auditable trace with no chain-of-thought.

**Met.** `TaskType.OPTICAL_SAR_FUSION` routes automatically for an optical+SAR pair; registry
entry `optical_sar_fusion` is `degraded` with real version/limitations; evidence includes a
fusion composite, complementary map, aligned previews, modality comparison, water agreement, and
regions, all served via the evidence-token endpoint; confidence is signal-derived; the trace
records modality resolution, co-registration, and the cross-modal analysis.

## Dependencies
Raster ingestion (03), registry (06), agent (05), evidence/confidence (12); Phase 6
co-registration (`align_modalities`). Libraries: NumPy, SciPy, Rasterio (`WarpedVRT`), pyproj.
No model weights, no GPU. Optional: CLIP (Phase 4/5) for the optical land-cover reading.

## Acceptance criteria
- [x] Joint optical+SAR result demonstrably uses both modalities; method documented.
- [x] Incompatible pairs rejected clearly; two-optical/two-SAR refused.
- [x] Complementary evidence (correlation, information partition, water agreement, regions).
- [x] Cross-modal VQA + honest limitations.
- [x] Evidence, evidence-derived confidence, auditable trace.
- [x] Phase 1–6 regression intact; ruff/eslint/tsc/build/compose green.

## Tests
`test_cross_modal.py` (unit: alignment, SAR/optical features, complementarity, water agreement,
rendering, against known synthetic truth) and `test_fusion_api.py` (integration: registry,
routing, specialist, negatives, evidence tokens, trace, regression). Fusion runs without
weights, so it is always exercised in CI; the optional land-cover reading is gated on
`models_ready`.

## Resolved risks
- ~~Genuine fusion is hard on CPU; may need a cloud GPU.~~ A deterministic feature-level method
  is genuine, CPU-only, and reproducible; a learned model can be added later behind the same
  interface.

## Outstanding (later phases)
- Benchmark cross-modal accuracy or document the absence of a suitable public benchmark — Phase 9.
- Confidence calibration — Phase 9. Status stays `degraded` until then.
- Learned fusion model; polarimetric SAR decomposition; sub-pixel co-registration — not scheduled.
