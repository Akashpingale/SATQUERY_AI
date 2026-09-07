# Spec 11 — Optical-SAR Fusion · Design (as built)

The design implemented in Phase 7. Method detail and measured behaviour: `docs/phase7.md`.

## Shape of the change

```
geospatial/resample.py      _plan_georeferenced_grid() extracted; align_modalities() added
                            (keeps all optical bands + all SAR bands on the shared grid)
ml/inference/fusion.py      analyze_cross_modal(): SAR/optical features, correlation/
                            complementarity, information partition, water agreement,
                            SAR-only structure regions; classify_cross_modal_question()
ml/inference/fusion_render.py  fusion composite, complementary map, aligned optical/SAR
        |
backend/app/agent/specialists/fusion_support.py  resolve modalities, align, analyze,
                            evidence assembly, measurement_score
backend/app/agent/specialists/fusion.py           OpticalSARFusionSpecialist (wording only)
backend/app/registry/models.py   optical_sar_fusion -> DEGRADED, version from FUSION_ALGORITHM_VERSION
backend/app/agent/understanding.py  cross-modal question terms added to fusion routing
frontend EvidencePanel/VisualizationPanel  cross-modal evidence renderers + Optical+SAR view
```

Layering holds: the analysis imports neither torch nor the web layer. The question classifier
lives in `ml.inference.fusion` so the same routing is reusable.

## Key design decisions

### Genuine feature-level fusion, not caption concatenation
A joint 2-channel representation (optical brightness + SAR backscatter on one grid) yields
cross-modal quantities no single modality produces: correlation/complementarity, the
optical-only / SAR-only / shared information partition, optical↔SAR water agreement (IoU), and
SAR-only structure regions. This is the R13 "genuine multimodal integration" requirement.

### Extract the grid planner rather than duplicate alignment
`_plan_georeferenced_grid` was factored out of the Phase 6 `_align_georeferenced` so `align_pair`
(temporal, min-band) and `align_modalities` (fusion, all bands both sides) share identical
co-registration geometry and trace operations. Phase 6 behaviour is unchanged (regression test
`test_align_pair_unaffected_by_the_refactor`).

### One task, one specialist
`TaskType.OPTICAL_SAR_FUSION` has a single tool `optical_sar_fusion`. Description and all
cross-modal VQA kinds are handled by one specialist branching on the question, because they read
the *same* measurement — mirroring how the temporal task splits are the exception, not the rule.
New task types were deliberately not added.

### SAR uncalibrated by default
Log10-scaled, robust-normalised backscatter; calibration claimed only when metadata declares it.
SAR is described by measured backscatter/roughness, never semantic classes.

### Learned component is optional and optical-only
CLIP land-cover reading of the optical image only, labelled indicative; never applied to SAR
(CLIP was not trained on radar). Absent weights degrade gracefully.

### Status DEGRADED, checkpoint None
Real and deterministic, not benchmarked. `checkpoint=None` = nothing to load.

## Honesty mechanisms
One-optical + one-SAR enforced; correlation withheld for a near-uniform modality; NDVI/NDWI and
water agreement need declared bands; coordinates only when georeferenced; m² only for metric
CRSs; every rendered pixel backed by a measured condition; confidence "Unable to determine" for
unanswerable questions; algorithm version `xmodal-1.0.0` in registry/trace/evidence.

## Configuration
Constants in `ml.inference.fusion` (`TEXTURE_WINDOW`, `STRUCTURE_TEXTURE`, `NDWI_WATER`,
`SAR_DARK_PERCENTILE`, `MIN_FEATURE_SPREAD`, `MIN_REGION_PIXELS`, `MAX_REGIONS`). No env-tunable
settings were needed; add `ParamSpec` entries later if agent-tuning is wanted.

## Not in scope
Learned fusion model; polarimetric decomposition; sub-pixel co-registration; benchmarking and
calibration (spec 14, Phase 9).
