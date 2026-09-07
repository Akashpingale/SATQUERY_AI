# Spec 01 — Foundation · Design

## Overview
A modular FastAPI backend, a React/TS frontend shell, and a metadata database, all
runnable locally (CPU, SQLite) and via Docker Compose (PostgreSQL). Structure and
contracts are chosen so later phases (raster, agent, models) attach cleanly.

## Backend structure
```
backend/
├── app/
│   ├── main.py                 # app factory: middleware, logging, exception handlers, routers
│   ├── core/
│   │   ├── config.py           # Pydantic Settings (env-driven)
│   │   └── logging.py          # structured logging + request id
│   ├── api/v1/
│   │   ├── router.py           # aggregates route modules under /api/v1
│   │   └── routes/
│   │       ├── health.py       # GET /health (app + db status)
│   │       ├── models.py       # GET /models (registry)
│   │       └── tasks.py        # GET /tasks (task registry)
│   ├── registry/
│   │   ├── tasks.py            # TaskType definitions + task registry
│   │   └── models.py           # ToolSpec entries + model/tool registry
│   ├── schemas/                # Pydantic response models
│   ├── db/
│   │   ├── base.py             # DeclarativeBase + metadata
│   │   └── session.py          # engine/session factory from DATABASE_URL
│   └── models_db/
│       └── analysis.py         # Analysis ORM model
├── alembic/                    # migration env + versions
├── tests/                      # pytest (SQLite, no GPU)
└── pyproject.toml
```

### Configuration (`core/config.py`)
Pydantic `Settings` reads `APP_ENV`, `LOG_LEVEL`, `LOG_FORMAT`, `API_V1_PREFIX`,
`CORS_ORIGINS`, `DATABASE_URL`, `STORAGE_DIR`, `MODELS_DIR`, `DATA_DIR`,
`MAX_UPLOAD_MB`, `DEVICE`, `APP_VERSION`. Defaults make the app runnable with no
`.env` (SQLite under `storage/`).

### Health (`GET /api/v1/health`)
Returns `{ status, app, version, environment, database: {connected, dialect}, device }`.
DB check executes `SELECT 1` and reports connected true/false (never throws to client).

### Registry-backed discovery
`GET /api/v1/models` and `GET /api/v1/tasks` return static-but-real registry contents.
In Phase 1 tools are declared with `status = "planned"` and honest metadata; later
phases flip them to `available` when a real implementation + checkpoint exists. This
gives the agent (Phase 3) a single source to select from and never invents tools.

### Logging & errors
`core/logging.py` configures JSON (prod) or console (dev) logging and injects a
per-request id via middleware. A global exception handler returns
`{ error: { code, message, request_id } }` — sanitized, no stack traces.

## Database
- `DATABASE_URL` selects the backend (SQLite file for dev/tests; `postgresql+psycopg`
  in Docker). Portable column types only (String, DateTime, Float, JSON, Text).
- `Analysis` table (Phase 1 shape): `id` (uuid str), `created_at`, `updated_at`,
  `status`, `task`, `query`, `model_id`, `model_version`, `confidence`,
  `confidence_level`, `input_metadata` (JSON), `evidence_refs` (JSON), `trace` (JSON),
  `error` (JSON), `report_ref`. Rasters/artifacts are NOT stored in the DB.
- Alembic migration `0001_initial` creates the table on both SQLite and PostgreSQL.

## Frontend structure
```
frontend/
├── src/
│   ├── main.tsx                # bootstraps Router + QueryClientProvider
│   ├── App.tsx                 # routes
│   ├── lib/api.ts              # shared fetch client (VITE_API_BASE_URL)
│   ├── lib/utils.ts            # cn() helper (shadcn-style)
│   ├── hooks/useHealth.ts      # TanStack Query hook -> GET /health
│   ├── components/
│   │   ├── ui/                 # shadcn-style primitives (button, badge, card)
│   │   ├── layout/Header.tsx   # branding + SystemStatus
│   │   ├── SystemStatus.tsx    # icon+text (not color-only) health indicator
│   │   └── panels/             # Input / Visualization / Assistant / History placeholders
│   └── pages/Workspace.tsx     # main layout composition
├── tests/                      # Vitest + RTL
└── package.json
```
- Strict TS. Tailwind + a small shadcn-style component set (added manually to avoid
  interactive CLI). TanStack Query for server state; single API client module.
- `SystemStatus` shows Online/Offline/Checking with both an icon and text label.

## Docker
- `docker/backend.Dockerfile`: python:3.11-slim, install backend, run uvicorn; healthcheck hits `/api/v1/health`.
- `docker/frontend.Dockerfile`: node build stage → nginx serve stage; healthcheck on `/`.
- `docker-compose.yml`: `postgres` (with healthcheck + named volume), `backend`
  (depends_on postgres healthy; runs migrations then serves), `frontend` (depends_on backend).

## Testing strategy
- `tests/test_health.py`: 200 + shape + db.connected true on SQLite.
- `tests/test_models.py` / `test_tasks.py`: registries non-empty, correct schema,
  statuses honest.
- FastAPI `TestClient`; DB points at a temp SQLite file. No GPU, no Postgres needed.

## Decisions & rationale
- **SQLite default** keeps dev/tests frictionless; Postgres only where it matters (Docker).
- **Registry-first** (even before models exist) enforces the "agent selects only from
  registry" rule from day one and prevents fake tool names later.
- **Manual shadcn-style setup** avoids non-deterministic interactive scaffolding.
