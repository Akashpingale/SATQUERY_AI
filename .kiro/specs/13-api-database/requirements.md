# Spec 13 — API & Database (stub)

> Phase 1 seeds this (health/models/tasks + Analysis table); expanded across phases.

## Introduction
Versioned FastAPI surface and PostgreSQL metadata store backing uploads, analysis,
trace, evidence, reports, and history.

## Requirements (high level)
- Endpoints: `GET /health`, `GET /models`, `GET /tasks`, `POST /uploads`,
  `POST /analysis`, `GET /analysis/{id}`, `/analysis/{id}/trace`, `/analysis/{id}/evidence`,
  `/analysis/{id}/report`, `GET /history`.
- Pydantic schemas + OpenAPI docs; sanitized errors; versioned under `/api/v1`.
- PostgreSQL stores metadata/trace/evidence-refs/provenance; rasters/artifacts on disk.
- Alembic migrations; SQLite for dev/tests, PostgreSQL in Docker.

## Design (outline)
`api/v1/routes/*`, `schemas/*`, `db/*`, `models_db/*`, `alembic/`. Storage service for
files under `storage/` with safe unique IDs.

## Implementation tasks (outline)
(Phase 1) health/models/tasks + Analysis table + migration. (Later) uploads → analysis
→ retrieval → trace/evidence/report → history, each with schemas + tests.

## Dependencies
Backend core (01); consumed by agent (05), specialists, evidence (12), frontend (02).

## Acceptance criteria
- Endpoints match schemas; migrations run on SQLite + Postgres; errors sanitized.

## Tests
API schema/contract tests; DB migration test; endpoint integration tests; negatives.

## User inputs required
- Postgres credentials via `.env` for Docker/prod (defaults provided for local).

## Known risks
- Keeping large binaries out of the DB; consistent id/filename safety.
