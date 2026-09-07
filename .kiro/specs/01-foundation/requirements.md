# Spec 01 — Foundation · Requirements

## Introduction
Phase 1 establishes a runnable, tested project skeleton: repository structure,
configuration, a FastAPI backend with health/models/tasks endpoints, a metadata
database (SQLite for dev/tests, PostgreSQL in Docker), a React/TypeScript frontend
shell that reports backend health, structured logging, and a reproducible Docker
setup. No AI is implemented in this phase; contracts and structure are put in place
so later phases plug in without rework.

## Requirements

### Requirement 1 — Project structure & configuration
**User story:** As a developer, I want a consistent, documented repository structure
and environment-based configuration, so that every phase has a clear home and no
secrets are hard-coded.

**Acceptance criteria**
1. WHEN the repository is inspected THEN it SHALL contain `backend/`, `frontend/`,
   `ml/`, `geospatial/`, `docker/`, `docs/`, `scripts/`, `data/`, `models/`,
   `storage/`, and `.kiro/`.
2. WHEN configuration is needed THEN the backend SHALL read it from environment
   variables (via `.env`) with documented defaults in `.env.example`.
3. IF a secret (token/key/password) is required THEN it SHALL only be provided via
   `.env`/secret manager and SHALL NOT be committed, logged, or bundled in frontend.
4. WHEN `.gitignore` is applied THEN `.env`, virtualenvs, `node_modules`, models,
   datasets, and runtime storage SHALL be excluded while `.env.example` is tracked.

### Requirement 2 — Backend API skeleton & health
**User story:** As a frontend/integrator, I want a versioned API with a health check,
so that I can verify the system is up and discover capabilities.

**Acceptance criteria**
1. WHEN the backend starts THEN it SHALL expose a FastAPI app under `/api/v1` with
   OpenAPI docs at `/docs`.
2. WHEN `GET /api/v1/health` is called THEN it SHALL return status, app version,
   environment, and database connectivity status.
3. WHEN `GET /api/v1/models` is called THEN it SHALL return the model/tool registry
   entries (id, task, modality, status).
4. WHEN `GET /api/v1/tasks` is called THEN it SHALL return the supported task types.
5. WHEN an unexpected error occurs THEN the API SHALL return a sanitized JSON error
   (no stack trace, no secrets) and log details server-side.

### Requirement 3 — Metadata database
**User story:** As the system, I want to persist analysis metadata, so that history,
status, and provenance can be retrieved.

**Acceptance criteria**
1. WHEN the app runs locally without a database server THEN it SHALL use SQLite by
   default (via `DATABASE_URL`).
2. WHEN the app runs under Docker Compose THEN it SHALL use PostgreSQL.
3. WHEN migrations are applied THEN an `analyses` table SHALL exist storing id,
   timestamps, task, status, selected model/tool, confidence, evidence refs, trace,
   error info, and input metadata.
4. WHEN the health endpoint runs THEN it SHALL report whether the database is reachable.

### Requirement 4 — Frontend application shell
**User story:** As a user/judge, I want a clean web app shell that shows system
status, so that the product is immediately approachable.

**Acceptance criteria**
1. WHEN the frontend loads THEN it SHALL render a header (branding + system status),
   and placeholders for input, visualization, assistant, and history panels.
2. WHEN the app mounts THEN it SHALL call `GET /api/v1/health` via the shared API
   client and display a non-color-only status indicator (icon + text).
3. IF the backend is unreachable THEN the UI SHALL show an accessible error state,
   not a blank screen or raw error.
4. WHEN built THEN the frontend SHALL compile with strict TypeScript and pass its tests.

### Requirement 5 — Logging & error handling
**User story:** As an operator, I want structured logs and safe errors, so that I can
debug without leaking sensitive data.

**Acceptance criteria**
1. WHEN the backend handles a request THEN it SHALL emit structured logs (JSON in
   production) including a request id, without secrets.
2. WHEN an error is returned to a client THEN it SHALL be sanitized.

### Requirement 6 — Reproducible Docker setup
**User story:** As an evaluator, I want to run the whole stack on a clean machine,
so that setup is trivial.

**Acceptance criteria**
1. WHEN `docker compose up --build` is run THEN it SHALL start `frontend`, `backend`,
   and `postgres` services with health checks.
2. WHEN services start THEN backend SHALL wait for a healthy database before serving.
3. WHEN the frontend container serves THEN it SHALL be reachable in a browser and
   able to reach the backend API.

## Dependencies
- Environment: Python 3.11+, Node 20+, Docker (verified in Phase 0).
- No external credentials or datasets required for this phase.

## User inputs required
- None for Phase 1. (Compute Path A already chosen: CPU dev now, cloud GPU later.)

## Known risks
- Windows C: drive low on space → keep venv/models/Docker data on D:.
- Rasterio/GDAL install can be finicky on Windows → deferred to Phase 2; not in the
  Phase 1 backend dependency set.
- Docker Desktop must be running for the clean-machine test.
