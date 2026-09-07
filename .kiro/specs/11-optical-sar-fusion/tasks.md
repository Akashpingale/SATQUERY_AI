# Spec 11 — Optical-SAR Fusion · Tasks

- [x] 1. Inspect Phases 1–6; confirm the fusion toolchain, registry stub, `_FUSION_TERMS`,
      `has_optical_sar_pair`, and feasibility filter already existed
- [x] 2. Design cross-modal fusion on the existing architecture; justify algorithmic
      feature-level fusion over a large multimodal model against the 4 GB constraint
- [x] 3. Extract `_plan_georeferenced_grid`; add `align_modalities` (all optical + all SAR bands
      on the shared grid); confirm `align_pair` unchanged (regression test)
- [x] 4. SAR preprocessing (linear-average → log10 → robust normalise) with uncalibrated-by-
      default representation and metadata-driven calibration detection
- [x] 5. Optical feature extraction (brightness, spectral variance, texture, NDVI/NDWI from
      declared bands only)
- [x] 6. `analyze_cross_modal`: correlation/complementarity, information partition, water
      agreement, SAR-only structure regions; withhold correlation for a uniform modality
- [x] 7. Fusion rendering: false-colour composite, complementary map, aligned optical/SAR
- [x] 8. `fusion_support` (resolve modalities, align, analyze, evidence, measurement_score) and
      the `OpticalSARFusionSpecialist` (describe + cross-modal VQA)
- [x] 9. Registry: flip `optical_sar_fusion` to DEGRADED, real version, honest limitations;
      register the specialist; add cross-modal question terms to routing
- [x] 10. Frontend: cross-modal evidence renderers (composite, complementary map, modality
      comparison, water agreement, complementary region) + an Optical+SAR visualization view
- [x] 11. Tests: `test_cross_modal.py` (unit, known-truth) + `test_fusion_api.py` (integration,
      negatives, evidence, trace, regression); update Phase 4 test that asserted fusion
      "unavailable"; update the planning test's unimplemented list
- [x] 12. Full verification: 404 backend / 96 frontend pass, ruff/eslint/tsc/build/compose green;
      performance measured (peak ~50 MB, sub-second per stage)

## Corrections made during the phase
- [x] `align_pair` truncates to min band count — wrong for fusion; added `align_modalities`
      keeping both band sets, via an extracted shared grid planner
- [x] Locale-independent number/area formatting reused for cross-modal region sizes

## Deliberately not done
- [ ] Learned fusion model — no open/licensed/fitting model that would not fabricate semantics
- [ ] Semantic SAR land-cover classification — refused; SAR reported as measured backscatter
- [ ] New task types for optical_evidence/sar_evidence — kept one task, rich evidence, to avoid
      a second architecture

## Remaining (later phases)
- [ ] Benchmark cross-modal accuracy or document the absence of a suitable public benchmark — Phase 9
- [ ] Confidence calibration — Phase 9; status stays `degraded` until then
- [ ] Polarimetric SAR decomposition; sub-pixel co-registration — not scheduled
