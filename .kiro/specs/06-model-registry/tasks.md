# Spec 06 — Model / Tool Registry · Tasks

- [x] 1. `TaskSpec` + `TASK_REGISTRY` (8 task types, image counts, modalities, phase)
- [x] 2. `ToolSpec` + `ParamSpec` + `ToolStatus`; all 10 logical tools declared
- [x] 3. Honest seeding: `planned` status, `checkpoint=None`, `license="TBD"` until chosen
- [x] 4. `Modality.UNKNOWN` so undetermined inputs are representable, not guessed
- [x] 5. Exposure via `GET /models` and `GET /tasks` with a health map
- [x] 6. Phase 2: `raster_preprocessor` + `registration_validator` → `available` v1.0.0
- [x] 7. Two independent runnability gates (registry status + registered implementation)
- [x] 8. Parameter allowlist enforcement (`validate_parameters`)
- [x] 9. Tests: registry contents, status honesty, toolchain registration, allowlist

## Remaining (per phase, as specialists land)
- [ ] Phase 4: `vqa`, `captioning`, `grounding` → implementation + `available`
- [ ] Phase 4: `evidence_aggregator`, `confidence_calibrator` formalized as tools
- [ ] Phase 6: `temporal_change`, `change_vqa`
- [ ] Phase 7: `optical_sar_fusion`
- [ ] Per-tool device requirements once real models are chosen (CPU vs GPU, VRAM)
- [ ] Health checks that verify a checkpoint is present and loadable
