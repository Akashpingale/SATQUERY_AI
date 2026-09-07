---
inclusion: always
---

# Architecture Rules

Keep the system **modular** with clear boundaries. No god-files, no tight coupling
of the UI to individual models.

## Layers (dependencies point downward only)
```
Frontend  ->  Backend API (/api/v1)  ->  Agent/Orchestrator
                                       ->  Geospatial ingestion
                                       ->  Model/Tool Registry  ->  AI/ML specialists
                                       ->  Evidence / Confidence / Trace
Backend  ->  PostgreSQL (metadata only)   +   filesystem/object storage (rasters, artifacts)
Training (ml/) is separable from serving and may run on a cloud GPU.
```

## Rules
- **Uniform specialist interface**: every model/tool (VQA, captioning, grounding,
  temporal, fusion) implements the same input/output contract so models are swappable.
- The **agent selects only from the registry**, with allowlisted, validated parameters.
  No arbitrary or model-invented parameters.
- The **UI depends on API contracts**, never on a specific model or checkpoint.
- **Heavy rasters live on disk / object storage**; PostgreSQL stores metadata,
  status, trace, evidence references, and provenance only.
- Backend organized by responsibility: `api/`, `agent/`, `registry/`, `raster/`,
  `evidence/`, `confidence/`, `trace/`, `db/`, `schemas/`, `core/`.
- Configuration comes from environment (`.env`), never hard-coded. Models are
  configurable/replaceable via the registry, not scattered `if` statements.
- Prefer async for expensive jobs; enforce timeouts and clean up temp artifacts.
- Keep the app **runnable after every meaningful phase**.
