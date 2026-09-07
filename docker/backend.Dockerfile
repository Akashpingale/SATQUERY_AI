# SatQuery AI backend image.
# Build context is the repository root (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 1. Shared libraries (they live at the repo root so the backend and the ML pipelines
#    share one implementation). Installed from an isolated build directory so their
#    pyproject.toml cannot collide with the backend's.
#    Rasterio wheels bundle GDAL, so no system GDAL packages are required.
COPY pyproject.toml README.md /build/satquery-core/
COPY geospatial /build/satquery-core/geospatial
COPY ml /build/satquery-core/ml
RUN pip install /build/satquery-core && rm -rf /build

# 2. Inference dependencies. Torch comes from the CPU wheel index: the default index
#    would pull ~2.5 GB of CUDA libraries that this deployment cannot use.
#    peft is included because loading the Phase 5 LoRA adapter requires it; without it
#    the container would silently serve only the baseline model.
ARG INSTALL_INFERENCE=true
RUN if [ "$INSTALL_INFERENCE" = "true" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch && \
        pip install "transformers>=4.44,<5" "safetensors>=0.4" "peft>=0.11"; \
    fi

# 3. Backend application. psycopg[binary] bundles libpq, so no extra system
#    packages are needed for PostgreSQL.
WORKDIR /app
COPY backend/pyproject.toml backend/README.md ./
COPY backend/app ./app
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./alembic.ini
RUN pip install ".[postgres]"

COPY docker/backend-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

# Reports healthy only when the API answers on the health endpoint.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health').read()" || exit 1

CMD ["/usr/local/bin/entrypoint.sh"]
