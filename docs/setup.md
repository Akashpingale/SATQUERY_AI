# Setup & Troubleshooting — SatQuery AI

## Prerequisites
- Python 3.11+ · Node.js 20+ · npm · Git · Docker Desktop (for containerized run)
- Disk: on this machine, prefer the **D:** drive (C: is low on free space)

## 1. Configure environment
```powershell
Copy-Item .env.example .env
# Edit .env: set POSTGRES_PASSWORD (for Docker). HF_TOKEN can stay blank - every model and
# dataset used so far is ungated.
```

## 2. Backend (local, CPU, SQLite)
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1

# Two installs: the shared libraries, then the API.
pip install -e .                 # satquery-core (geospatial + ml)
pip install -e "backend[dev]"    # FastAPI app + pytest/ruff

# Inference dependencies (Phase 4 single-image AI). Torch MUST come from the CPU index:
# the default index pulls ~2.5 GB of CUDA wheels this machine cannot use.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.44,<5" "safetensors>=0.4"

# apply migrations (creates analyses + raster_assets)
alembic -c backend/alembic.ini upgrade head

# run tests
cd backend; pytest; cd ..

# run API
uvicorn app.main:app --app-dir backend --reload --port 8000
# -> http://localhost:8000/docs  and  http://localhost:8000/api/v1/health
```
> `geospatial/` lives at the repository root (per the master plan layout) so the backend
> and the ML pipelines share one implementation — hence the separate `pip install -e .`.

> To keep the virtualenv off C:, create it on D: (e.g. `python -m venv D:\satquery-venv`)
> and activate that path instead.

## 3. Frontend (local)
```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run test       # vitest
npm run build      # production build
```
Set `VITE_API_BASE_URL` in `.env` (defaults to `http://localhost:8000/api/v1`).

## 4. Full stack with Docker (clean machine)
```powershell
Copy-Item .env.example .env   # set POSTGRES_PASSWORD
docker compose up --build
# Frontend: http://localhost:8080
# API:      http://localhost:8000/docs
```
The backend waits for a healthy PostgreSQL, runs migrations, then serves.

## Database
- Local dev/tests use **SQLite** (no server needed).
- Docker uses **PostgreSQL**. Apply migrations: `alembic -c backend/alembic.ini upgrade head`
  (run automatically in the backend container start command).

## Troubleshooting
- **C: drive full / Docker fails to build**: move Docker Desktop's disk image to D:
  (Settings → Resources → Advanced), and keep models/datasets under D:.
- **`torch`/GPU**: serving defaults to `DEVICE=cpu` and needs no GPU. Phase 5 training does
  require CUDA and **refuses to run on CPU** rather than misreporting the result. Install the
  CUDA wheel (`pip install torch --index-url https://download.pytorch.org/whl/cu124`) and prove
  the device works with `python scripts/verify_phase5_gpu.py` before training. Training was
  performed on an NVIDIA RTX 2050 (4 GB); a phantom "RTX 3050 6GB" also appears in Windows
  Device Manager on this machine in error state Code 45 and is not usable.
- **CUDA out of memory during training**: `scripts/train_phase5.py` probes the batch size
  downward automatically. Batch 32 fits in 4 GB (peak 2,219.2 MB measured).
- **Adapted model not being used**: check `PHASE5_ADAPTER_DIR` points at a directory
  containing `adapter_config.json` and `adapter_model.safetensors`, and that `peft` is
  installed. When the adapter cannot load, the backend serves the Phase 4 baseline and the
  execution trace states why — so read the trace rather than guessing.
- **Rasterio/GDAL (Windows)**: the published rasterio wheels bundle GDAL, so no system
  GDAL install is needed. Verified working with rasterio 1.4.4 / GDAL 3.10.3. If a wheel
  is unavailable for your Python version, use the Docker path instead.
- **`ModuleNotFoundError: geospatial` or `ml`**: run `pip install -e .` from the repository
  root (both ship in the `satquery-core` distribution).
- **Model weights / disk space**: the Phase 4 baselines download ~1.2 GB on first use. Set
  `HF_HOME` to a drive with space (e.g. `D:\SatQuery-AI\models\hf-cache`). Verify with
  `python scripts/check_models.py`, which reports real load and inference times.
- **Analyses return `unavailable` with `model_unavailable`**: the weights are not cached and
  downloads are disabled. Unset `HF_HUB_OFFLINE` or pre-download via `scripts/check_models.py`.
- **`transformers` 5.x**: pinned to `<5`. The 5.x line was not stable in this environment.
- **Slow first analysis**: the first request loads the model and encodes concept prompts
  (~3 s). Subsequent requests reuse both caches (~0.06 s for scoring).
- **Tests never download models**: `conftest.py` forces offline mode, and model-dependent
  tests skip when weights are absent.
- **Storage paths**: relative paths in `.env` resolve against the repository root, not the
  current working directory, so the app behaves the same launched from anywhere.
- **Port conflicts**: change `API_PORT` / Vite port / mapped compose ports.
- **CORS errors**: ensure the frontend origin is in `CORS_ORIGINS`.

## Verifying a healthy setup
```powershell
# backend
curl http://localhost:8000/api/v1/health
# expect: status ok, database.connected true
```
