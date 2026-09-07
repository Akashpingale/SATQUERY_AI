# Spec 10 — Temporal Change · Tasks

- [x] 1. Inspect Phases 1–5; confirm the temporal toolchain, registry stubs, and T1/T2 upload
      slots already existed, and that no co-registration function did
- [x] 2. Select the change-detection approach against the 4 GB constraint; document why a
      pretrained CD network was rejected (`docs/phase6.md`)
- [x] 3. `geospatial/resample.py`: `align_pair` co-registers a pair onto one grid via WarpedVRT
      (reproject, coarser-resolution resample, overlap crop, memory cap), recording operations
- [x] 4. `geospatial/bands.py`: declared-only band-role resolution (no guessing) + normalized
      difference helper
- [x] 5. `ml/inference/temporal.py`: change vector analysis, SAR log-ratio, robust two-pass
      normalisation, noise-floor threshold, coherence filtering, misregistration gate, NDVI/NDWI
- [x] 6. `ml/inference/change_map.py`: change map, T1/T2/change strip, region crop for the
      optional land-cover reading
- [x] 7. `temporal_support.py`: one measurement, evidence assembly, `measurement_score`, trace
      notes; add `SpecialistResult.trace_notes` and emit it from the orchestrator
- [x] 8. `TemporalChangeSpecialist` (grounded description) reading measured signals
- [x] 9. `ChangeVQASpecialist` answering occurrence/where/magnitude/vegetation/water/built-up/
      which-date; honest limitation for unsupported questions
- [x] 10. Registry: flip `temporal_change` + `change_vqa` to DEGRADED, real version from
      `ALGORITHM_VERSION`, allowlisted threshold parameters, honest limitations
- [x] 11. Frontend: change_map/temporal_compare/change_region/indices/semantic evidence
      renderers; T1/T2 compare + change views in the visualization panel; locale-independent
      number/area formatters
- [x] 12. `ml/evaluation/change_vqa.py`: CDVQA-ready interface + coverage-aware metrics; no
      benchmark claims
- [x] 13. Tests: `test_change_detection.py` (unit, known-truth) + `test_temporal_api.py`
      (integration, negatives, eval adapter); update Phase 4 tests that asserted temporal was
      "unavailable"
- [x] 14. Full verification: 345 backend / 85 frontend pass, ruff/eslint/tsc/build/compose green

## Corrections made during the phase
- [x] Two-pass robust normalisation (a real NIR collapse had been normalised away)
- [x] Robust noise floor as the threshold criterion, Otsu reported-only (a dominant change had
      hidden a weaker one)
- [x] Carry the effective magnitude method out of the measurement pass (SAR mislabel)
- [x] `all_present()` instead of `None not in (arrays)` (array truth-value error)
- [x] `shift_explains` discriminator so a large change is not mistaken for misregistration
- [x] Distinct `change_region` evidence kind so georeferenced regions show coordinates
- [x] Added `scipy>=1.11` to `pyproject.toml`

## Deliberately not done
- [ ] Semantic change-type classification / built-up detection — no classifier trained; the
      system refuses the claim rather than fabricating it
- [ ] Optical–SAR temporal comparison — refused (different physical quantities); fusion is
      spec 11, Phase 7
- [ ] A pretrained deep change-detection model — rejected on resolution/domain mismatch

## Remaining (later phases)
- [ ] Benchmark change-detection quality (LEVIR-CD/OSCD) and run the CDVQA interface — Phase 9
- [ ] Confidence calibration against labelled change data — Phase 9
- [ ] Move registry status off `degraded` once accuracy is established — Phase 9
