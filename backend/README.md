# SatQuery AI — Backend

FastAPI service for the agentic remote-sensing analysis API. See the repository root
`README.md` for the full project overview and `docs/architecture.md` for design.

## Layout
```
app/
├── main.py        app factory (middleware, logging, exception handling, routers)
├── core/          config (env-driven) + structured logging
├── api/v1/        versioned routes (health, models, tasks; more per phase)
├── registry/      task registry + model/tool registry (agent selects only from here)
├── schemas/       Pydantic response models
├── db/            declarative base + engine/session (SQLite dev / Postgres docker)
└── models_db/     ORM models (Analysis)
alembic/           migrations
tests/             pytest (SQLite, no GPU, no running Postgres)
```

## Develop
```bash
pip install -e ".[dev]"        # local dev/tests (SQLite)
pip install -e ".[dev,postgres]"  # add the PostgreSQL driver
pytest                          # run tests
uvicorn app.main:app --reload   # serve (run from this directory)
alembic -c alembic.ini upgrade head   # apply migrations
```

Endpoints (Phase 1): `GET /api/v1/health`, `GET /api/v1/models`, `GET /api/v1/tasks`.
OpenAPI docs at `/docs`.
