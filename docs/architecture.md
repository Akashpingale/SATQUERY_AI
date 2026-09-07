# Architecture — SatQuery AI

Companion to `.kiro/steering/architecture.md` (binding rules) and the master plan.
This document describes the concrete system as it is built out.

## System overview
```
┌──────────────────────────────────────────────────────────────┐
│  Frontend (React + TS + Vite + Tailwind + shadcn/ui)           │
│  Upload · Viewer · Query · Answer · Evidence · Confidence ·    │
│  Execution Trace · History · Reports                           │
└───────────────▲───────────────────────────────┬───────────────┘
                │ REST /api/v1 (TanStack Query)   │
┌───────────────┴───────────────────────────────▼───────────────┐
│  Backend API (FastAPI + Pydantic)                              │
│  health · models · tasks · uploads · analysis · trace ·        │
│  evidence · report · history   ·   OpenAPI · structured logs   │
└──┬───────────┬───────────────┬───────────────┬────────────────┘
   ▼           ▼               ▼               ▼
 Agent /    Geospatial     Model/Tool      Evidence +
 Orchestr.  Ingestion      Registry        Confidence + Trace
   │           │               │               │
   ▼           ▼               ▼               ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  AI/ML Inference (specialists behind a uniform contract)       │
 │  VQA · Captioning · Grounding · TemporalChange · ChangeVQA ·   │
 │  OpticalSARFusion   ← swappable (base + LoRA adapters)         │
 └───────────────▲───────────────────────────────┬───────────────┘
                 │                                 │
       Training/Adaptation (ml/, CUDA GPU)    Storage (uploads/
       LoRA · BigEarthNet.txt · retrieval     processed/evidence/
       eval  → models/phase5/adapter          reports/temp) + models/
 ┌──────────────────────────────────────────────────────────────┐
 │  PostgreSQL — analysis metadata, status, model/version,        │
 │  confidence, evidence refs, trace, provenance                  │
 └──────────────────────────────────────────────────────────────┘

 Packaging: Docker Compose → frontend · backend · postgres · (opt) redis/worker
```

## Boundaries (enforced)
- Dependencies point downward only (see steering `architecture.md`).
- Specialists share a uniform input/output contract; the agent selects only from the
  registry with allowlisted parameters; the UI depends on API contracts, not models.
- Rasters/artifacts live on the filesystem/object storage; PostgreSQL holds metadata only.
- `ml/` training is separable from serving: the backend imports only the adapter loader, so
  training can run on any GPU host (local or cloud) without changing the API.

## Backend layout (through Phase 3)
```
backend/app/
├── main.py            app factory (middleware, exception handlers, routers)
├── core/              config (env, repo-root anchored paths), structured logging
├── api/v1/            router + routes (health, models, tasks, uploads, analysis)
├── agent/             the orchestration pipeline
│   ├── types.py         pipeline contracts
│   ├── understanding.py query sanitization + task classification
│   ├── planner.py       task -> ordered registered tools
│   ├── validation.py    plan/parameter validation, runnability gate
│   ├── confidence.py    signal-derived confidence
│   ├── trace.py         observable execution trace
│   ├── orchestrator.py  run_analysis()
│   └── specialists/     uniform Specialist interface + implementations
├── registry/          task registry + model/tool registry (honest statuses)
├── schemas/           Pydantic request/response models
├── security/          filename sanitization, path confinement, content-type allowlist
├── services/          storage (streaming, size-limited) + raster ingestion
├── db/                declarative base + session (DATABASE_URL)
└── models_db/         Analysis, RasterAsset ORM models

geospatial/            shared raster library (part of the satquery-core distribution)
├── types.py  metadata.py  io.py  validation.py  alignment.py  render.py  coords.py
├── resample.py  align_pair(): co-register a pair onto one grid (WarpedVRT), Phase 6
└── bands.py     resolve_band_roles(): declared-only band identification for indices

ml/                    machine-learning library (same distribution, separable from serving)
├── inference/
│   ├── base.py               lazy loading, caching, device + availability reporting
│   ├── vocabulary.py         remote-sensing concepts, prompts, question analysis
│   ├── clip_backend.py       zero-shot concept scoring + coarse patch localization
│   ├── detection_backend.py  open-vocabulary detection
│   ├── overlay.py            box rendering for visual evidence
│   ├── temporal.py           Phase 6 change detection (algorithmic, CPU, no weights)
│   └── change_map.py         Phase 6 change-map + T1/T2/change strip rendering
└── evaluation/
    ├── retrieval.py          Phase 5 image-text retrieval metrics
    └── change_vqa.py         Phase 6 CDVQA-ready interface (not benchmarked)
```

## Temporal change flow (Phase 6)
```
T1 + T2 assets
  -> registration_validator            (compatible? overlap/CRS/resolution)  [stops if not]
  -> temporal_change / change_vqa specialist
       -> geospatial.align_pair         common grid; records reprojection/resample/crop
       -> ml.inference.detect_change    CVA/log-ratio + robust threshold + coherence
       -> optional CLIP land-cover read of the largest changed region (indicative)
       -> change map + T1/T2/change strip + per-region evidence (served by evidence token)
  -> evidence-based confidence + auditable trace (co-registration + measurement recorded)
```
Change detection is algorithmic and CPU-only, so it needs no GPU and runs in CI. The optional
land-cover reading is the only learned step and degrades gracefully when weights are absent.

## Inference layer (Phase 4)
```
raster -> load_display_image()  decimated RGB, tracks the scale factor
       -> CLIP (clip-rsicd-v2)  concept probabilities  -> VQA, captioning
       -> OWL-ViT               bounding boxes         -> grounding
                               |
                               +-> overlay PNG (visual evidence)
                               +-> box_to_geo(): pixels -> source pixels -> CRS -> lon/lat
```

Models are loaded lazily and cached for the process lifetime; concept **text** embeddings are
cached separately because they depend only on the vocabulary, not the image (warm scoring
~0.06 s versus ~3 s cold). Weights live in the Hugging Face cache (`HF_HOME`), never in the
image or the repository.

Two independent gates still govern execution: registry status and a registered
implementation. A capability whose weights cannot be loaded raises `ModelUnavailableError`,
which the orchestrator turns into an `unavailable` outcome — never a fabricated answer.

Capability status is deliberately `degraded` for the Phase 4 baselines: real inference, but
zero-shot, uncalibrated, and not yet adapted. See `docs/models.md`.

## Agent pipeline (Phase 3)
```
query + asset ids
   -> input inspection      (assets -> InputConfiguration)
   -> query understanding   (sanitize; injection fragments stripped + reported)
   -> task classification   (wording + input shape, feasibility-filtered)
   -> workflow planning     (task -> ordered REGISTERED tool ids)
   -> plan validation       (task shape, tool registration, asset refs, params, timeout)
   -> execution             (per step: runnability gate -> specialist.run())
   -> evidence -> confidence -> answer synthesis -> execution trace
```

Two independent gates make a tool runnable: its registry `status`, and a registered
implementation in `agent/specialists`. Declaring a tool can therefore never make it appear
to work. Tasks whose specialists are unbuilt return status `unavailable` with the delivering
phase named — never a substitute answer.

Runnability is checked **per step**, not per plan, so runnable earlier steps still report
their findings. That is why an incompatible temporal pair is told "these images do not
overlap" rather than "that specialist arrives in phase 6".

## Shared libraries
`geospatial/` and `ml/` sit at the repository root so the backend API and the ML pipelines
use one implementation. Both ship in the `satquery-core` distribution installed from the root
`pyproject.toml` (`pip install -e .`); inference dependencies are the optional `[inference]`
extra so the geospatial layer stays installable without ~500 MB of ML wheels.
Key guarantee: **all reads are windowed or decimated**, so large GeoTIFFs never load
fully into RAM. Previews are capped at 1024 px and the full-resolution raster is never
sent to the browser.

## Data model (through Phase 2)
- **`analyses`** — `id, created_at, updated_at, status, task, query, model_id,
  model_version, confidence, confidence_level, input_metadata(JSON), evidence_refs(JSON),
  trace(JSON), error(JSON), report_ref`.
- **`raster_assets`** — identity, original + stored filename, storage/preview paths
  (relative to the storage root), size, content type, role (`single|optical|sar|t1|t2`),
  `modality` (user-declared) and `detected_modality` (inferred hint, kept separate),
  dimensions/bands/dtype/driver, CRS + EPSG, `is_georeferenced`, full
  `raster_metadata`(JSON), ingestion `warnings`(JSON).

Raster bytes and previews live on the filesystem under the storage root; only metadata
is stored in the database.

## Ingestion flow (Phase 2)
```
upload -> sanitize filename -> stream to disk (size-limited)
       -> validate as real raster (extension, open, sanity limits)
       -> on failure: delete file, persist nothing, return structured issues
       -> on success: extract metadata, render preview, persist RasterAsset
pair   -> check_pair_compatibility (CRS reprojection, overlap, resolution, grid)
       -> verdict decided on the server, consumed by both the UI and the agent
```

## Environments
- **Local dev**: CPU, SQLite (`DATABASE_URL=sqlite:///...`), uvicorn + Vite.
- **Docker**: frontend (nginx) + backend (uvicorn) + PostgreSQL, with health checks. The
  backend image includes `peft`; the adapter is mounted via the `models` volume, not baked in.
- **GPU host** (training only): CUDA required, never used by the serving path. Artifacts flow
  back as a LoRA adapter directory located by `PHASE5_ADAPTER_DIR` and exposed as its own
  registry entry. Phase 5 ran on a local RTX 2050 (4 GB); a cloud GPU works identically.

## Evolution
Per master-plan phases 2–10: raster pipeline → agent → single-image → RS adaptation →
temporal → optical-SAR → UX polish → evaluation → finalization. This document is updated
each phase.
