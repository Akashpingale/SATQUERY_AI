# Phase 13 — Docker Release Smoke Test

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07

## Status

- `docker compose config` → **VALID / PASS** (exit 0).
- `docker compose build` + `up` full smoke → **NOT_COMPLETED (BLOCKED)** — the Docker daemon is not running in this environment.

Do **not** claim runtime success from `config` alone. The static configuration is proven valid; container startup was not exercised here.

## Evidence

| Command | Result |
|---|---|
| `docker compose --project-directory d:\SatQuery-AI config --quiet` | exit 0 (valid) |
| `docker compose config --services` | `postgres`, `backend`, `frontend` |
| `docker info` / `docker version --format {{.Server.Version}}` | **daemon unreachable** — `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified` |

Because `docker compose config` is a static parse, it succeeds without a daemon; `build`/`up` require the daemon and therefore could not run.

## Configuration inspected (static, from compose + Dockerfiles)

**Services & topology**
- `postgres` (postgres:16-alpine): named volume `pgdata`, `pg_isready` healthcheck (10s/5s/10 retries), port 5432.
- `backend` (`docker/backend.Dockerfile`): `APP_ENV=production`, `LOG_FORMAT=json`, `DEVICE=cpu`, `DATABASE_URL=postgresql+psycopg://…@postgres:5432/…`, `CORS_ORIGINS=http://localhost:8080`, storage/models on named volumes, port 8000. `depends_on: postgres (service_healthy)`.
- `frontend` (`docker/frontend.Dockerfile`): nginx serving the built SPA, port 8080→80, `depends_on: backend (service_healthy)`.

**Healthchecks**
- Backend: `HEALTHCHECK` in the Dockerfile hits `/api/v1/health` (15s interval, 40s start-period, 5 retries) → enables the frontend's `service_healthy` gate.
- Frontend: `HEALTHCHECK` wgets `/`.

**Startup dependencies / migrations**
- `backend-entrypoint.sh`: runs `alembic -c alembic.ini upgrade head` then `uvicorn app.main:app --host 0.0.0.0 --port 8000`. So the schema is migrated on container start.

**Image hygiene**
- Torch installed from the **CPU** wheel index (avoids ~2.5 GB CUDA libs).
- Model weights are **not baked** into the image; cached on the `models` volume (`HF_HOME=/app/models/hf-cache`) and downloaded on first use. The Phase 5 adapter is mounted via the volume; without it the backend serves the baseline and records the fallback (never fabricates "adapted").
- `.dockerignore` present.

**Volumes / network:** named volumes `pgdata`, `storage`, `models`; default compose network; ports 5432/8000/8080.

## Why full build/up matters and what remains

The compose graph is coherent (healthcheck-gated startup, migrations on entry, CPU-only torch, weights on a volume). The one thing not exercised here is that the images actually **build and start** on a clean daemon. Recommended (P1/P2) on a machine with Docker running:

```powershell
docker compose --project-directory d:\SatQuery-AI config            # already VALID
docker compose --project-directory d:\SatQuery-AI build              # ~large: torch + transformers
docker compose --project-directory d:\SatQuery-AI up -d
# then verify:
#  - postgres healthy (pg_isready)
#  - backend healthy at http://localhost:8000/api/v1/health
#  - frontend reachable at http://localhost:8080 and proxying /api to backend
#  - first analysis downloads weights to the models volume (not a startup crash)
docker compose --project-directory d:\SatQuery-AI down
```

**Note:** first run downloads model weights (~1.2 GB) to the volume; on a throttled link this is slow (see the VRSBench blocker for measured throughput). Budget time accordingly. No destructive Docker cleanup was performed.
