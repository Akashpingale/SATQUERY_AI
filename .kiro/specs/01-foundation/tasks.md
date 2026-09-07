# Spec 01 — Foundation · Tasks

- [x] 0. Environment inspection (Phase 0) and repo scaffold
  - git init; `.gitignore`, `.gitattributes`, `.env.example`, `LICENSE`, `README.md`
  - directory tree + `.gitkeep`s; `ml/` and `geospatial/` README stubs
  - _Reqs: 1_

- [x] 1. Kiro configuration
  - 9 steering files; 17 spec folders; `docs/traceability.md`
  - _Reqs: 1_

- [ ] 2. Backend core
  - [ ] 2.1 `core/config.py` (Pydantic Settings from env, safe defaults)
  - [ ] 2.2 `core/logging.py` (structured logging + request-id middleware)
  - [ ] 2.3 `main.py` app factory (CORS, exception handlers, router mount)
  - _Reqs: 2, 5_

- [ ] 3. Registry
  - [ ] 3.1 `registry/tasks.py` — TaskType enum + task registry
  - [ ] 3.2 `registry/models.py` — ToolSpec + model/tool registry (honest statuses)
  - _Reqs: 2_

- [ ] 4. API v1
  - [ ] 4.1 `routes/health.py` (+ DB `SELECT 1` check)
  - [ ] 4.2 `routes/models.py`, `routes/tasks.py`
  - [ ] 4.3 `api/v1/router.py` aggregation
  - _Reqs: 2, 3_

- [ ] 5. Database
  - [ ] 5.1 `db/base.py`, `db/session.py` (engine from DATABASE_URL)
  - [ ] 5.2 `models_db/analysis.py` (Analysis ORM, portable types)
  - [ ] 5.3 Alembic env + `0001_initial` migration (SQLite + Postgres)
  - _Reqs: 3_

- [ ] 6. Backend packaging + tests
  - [ ] 6.1 `pyproject.toml` (core deps + dev extras, ruff, pytest)
  - [ ] 6.2 `tests/` — health, models, tasks
  - _Reqs: 2, 3, 5_

- [ ] 7. Frontend shell
  - [ ] 7.1 Vite + React + TS + Tailwind + shadcn-style primitives
  - [ ] 7.2 Router + QueryClient + `lib/api.ts` + `useHealth`
  - [ ] 7.3 Header + SystemStatus (icon+text) + panel placeholders + Workspace page
  - [ ] 7.4 Vitest + RTL test for SystemStatus
  - _Reqs: 4_

- [ ] 8. Docker
  - [ ] 8.1 backend + frontend Dockerfiles
  - [ ] 8.2 `docker-compose.yml` (frontend/backend/postgres, healthchecks, volumes)
  - _Reqs: 6_

- [ ] 9. Verify
  - [ ] 9.1 backend: venv on D:, install, `pytest`
  - [ ] 9.2 frontend: `npm install`, build, `vitest run`
  - _Reqs: 2, 3, 4_

> Checkboxes are updated as implementation lands. Docker clean-machine run is
> validated during Phase 10 finalization if not fully exercised here.
