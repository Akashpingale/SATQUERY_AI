# Spec 16 — Docker & Deployment (stub)

> Phase 1 seeds compose; clean-machine validation at Phase 10.

## Introduction
Reproducible containerized setup for frontend, backend, and PostgreSQL (optional
Redis/worker later), runnable on a clean machine.

## Requirements (high level)
- Dockerfiles for backend and frontend; `docker-compose.yml` with frontend/backend/postgres.
- Health checks, environment configuration, and persistent volumes.
- Backend waits for a healthy DB and runs migrations before serving.
- Documented clean-machine commands; mindful of low C: space (use D: for Docker data).

## Design (outline)
`docker/backend.Dockerfile` (python:3.11-slim + uvicorn), `docker/frontend.Dockerfile`
(node build → nginx serve), `docker-compose.yml` (+ named volumes, healthchecks).

## Implementation tasks (outline)
(Phase 1) Dockerfiles + compose + healthchecks. (Phase 10) clean-machine test, optional
Redis/worker, CI Docker build + smoke test.

## Dependencies
Backend (01/13), frontend (02). Docker Desktop running.

## Acceptance criteria
- `docker compose up --build` brings up a reachable UI + API + DB with healthchecks.

## Tests
Compose smoke test (health endpoints reachable); CI docker build.

## User inputs required
- Postgres password via `.env`; deployment target/domain/HTTPS if publicly hosted (later).

## Known risks
- C: disk space for Docker images; Compose version quirks — validate early.
