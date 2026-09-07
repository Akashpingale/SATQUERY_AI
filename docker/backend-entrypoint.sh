#!/bin/sh
# Apply database migrations, then start the API server.
set -e

echo "[entrypoint] Applying database migrations..."
alembic -c alembic.ini upgrade head

echo "[entrypoint] Starting SatQuery AI API on :8000"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
