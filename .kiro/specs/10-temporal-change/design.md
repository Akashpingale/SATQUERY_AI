# Spec 10 — Temporal Change · Design (as built)

The design actually implemented in Phase 6, including the decisions reversed after
measurement. Method detail and measured behaviour: `docs/phase6.md`, `docs/evaluation.md`.

## Shape of the change

```
geospatial/resample.py      align_pair(): WarpedVRT read of both dates onto one grid
geospatial/bands.py         resolve_band_roles(): declared-only band identification
ml/inference/temporal.py    detect_change(): CVA/log-ratio + robust threshold + coherence;
                            classify_change_question(); magnitude_band(); ChangeConfig
ml/inference/change_map.py  render_change_map / render_comparison_strip / crop_pair
        |
backend/app/agent/specialists/temporal_support.py  analyze_pair(): one measurement,
                            evidence assembly, measurement_score(), trace notes
backend/app/agent/specialists/temporal.py           TemporalChangeSpecialist,
                            ChangeVQASpecialist (wording only)
backend/app/registry/models.py   temporal_change + change_vqa -> DEGRADED, params, version
ml/evaluation/change_vqa.py      CDVQA-ready interface (not run)
```

Layering holds: `backend` calls `ml.inference` and `geospatial`; the detector imports neither
torch nor the web layer. The question classifier lives in `ml.inference.temporal`, not the
specialist, so the evaluation harness routes questions exactly as the serving path does.

## Key design decisions

### Algorithmic change vector analysis, not a pretrained CD network
Pretrained BIT/ChangeFormer/TinyCD checkpoints are trained on ~0.5 m aerial building change and
produce confident, wrong masks at other resolutions/land-cover — the failure this project
forbids. They also ship as loose per-repo files. CVA measures a real quantity at any resolution,
reports its own uncertainty, needs no weights or GPU, and runs in CI. A learned model can be
added later behind the same specialist interface. Full rationale in `docs/phase6.md`.

### Co-register to the coarser resolution
`align_pair` takes the coarser of the two pixel sizes so the finer image is downsampled rather
than the coarser one being upsampled into detail it does not have. Reprojection and resampling
go through `WarpedVRT` (warp-on-read, bounded memory) and pyproj — never hand-rolled.

### Robust statistics throughout
Radiometric normalisation and the noise floor use median / MAD-scaled sigma, not mean / std,
because real change is an outlier population that mean-based statistics let bias the fit.

### Separate `change_region` evidence kind
Distinct from Phase 4 `region`: the payload carries magnitude/brightness/area rather than
detector scores, and reusing `region` made the UI render "coordinates unavailable" for a
georeferenced result.

### `trace_notes` on `SpecialistResult`
A step's internal operations (co-registration, registration check, measurement) must reach the
trace, which previously only carried the step's answer. `SpecialistResult.trace_notes` lets a
specialist emit observable steps; the orchestrator appends them in execution order. This is what
satisfies the geospatial "resampling recorded in the trace" rule.

### Status DEGRADED, checkpoint None
Real and deterministic, but not benchmarked (Phase 9). `checkpoint=None` means "nothing to
load", not "missing". `training_dataset`/`training_method` say "none - algorithmic".

## Bugs found by measurement (and fixes)

| Problem | Root cause | Fix |
|---|---|---|
| An obvious NIR collapse over 17% of a scene detected as 0% change | mean/std normalisation let the change bias its own fit | Two-pass robust normalisation: refit on presumed-unchanged pixels using median/MAD |
| A dominant bright change hid a genuine weaker change | `max(otsu, floor)` — Otsu's split climbs to separate the dominant change from everything else | Robust noise floor is the criterion; Otsu is reported but not used as the threshold |
| A SAR pair could report `log_ratio` while computing change-vector | the two-pass `measure()` dropped the method fallback from `change_magnitude` | carry the effective method out of the pass |
| `None not in (arrays)` raised "ambiguous truth value" | element-wise comparison against NumPy arrays | explicit `all(x is not None ...)` |
| A large real change was flagged as misregistration | any shift correlates slightly better | `shift_explains` discriminator: a shift must explain most of the residual decorrelation |
| `tsc -b` emitted a stale config file (Phase 5 fix, reconfirmed) | — | unchanged |

## Honesty mechanisms
Tint only over measured pixels (asserted by test); misregistration and optical-vs-SAR refused;
NDVI/NDWI only from declared bands; cause never asserted (no built-up classifier); area in m²
only for metric CRSs; confidence "Unable to determine" for unanswerable questions; every
threshold a named constant; algorithm version in registry/trace/evidence.

## Configuration
`ChangeConfig` holds every threshold (`noise_sigma`, `opening_size`, `min_region_pixels`,
`max_tolerable_shift_pixels`, `min_changed_fraction`, `suspicious_changed_fraction`,
`min_valid_fraction`, `low_correlation`). Three are exposed as allowlisted agent parameters
(`noise_sigma`, `min_region_pixels`, `min_changed_fraction`) with validated ranges.

## Not in scope
Optical–SAR fusion (spec 11, Phase 7). Benchmarking and calibration (spec 14, Phase 9). Learned
change models and multi-temporal analysis (not scheduled).
