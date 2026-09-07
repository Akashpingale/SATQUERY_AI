# SatQuery AI

**An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

- Problem: **SIH26167 — SatQuery AI**
- Organization: **ISRO / Department of Space**
- Category: Software · Theme: Space Technology

SatQuery AI is an **agentic** remote-sensing analysis assistant. A user uploads
satellite imagery (optical, SAR, temporal pairs, or co-registered optical+SAR),
asks a question in plain language, and the system understands the query,
validates the inputs, selects the right specialist models/tools, runs the
analysis, and returns an **answer with visual evidence, a confidence level, and
a concise execution trace**.

> This is a real end-to-end system, not a chatbot or a frontend mock. Mandatory
> remote-sensing capabilities are implemented, integrated, and tested. See
> `SatQuery_AI_Kiro_Master_Plan.md` (the engineering source of truth) and
> `docs/traceability.md`.

---

## Status

Built phase-by-phase. Honest current state:

| Phase | Scope | Status |
|---|---|---|
| 1 | Repo, backend, frontend shell, database, Docker | Done |
| 2 | Raster ingestion: GeoTIFF/TIFF + PNG/JPEG upload, validation, metadata, previews, pair compatibility | Done |
| 3 | Agent foundation: query understanding, registry-constrained routing, parameter validation, evidence/confidence/execution trace | Done |
| 4 | Single-image AI: VQA, captioning, grounding with real CPU inference and visual evidence | Done |
| 5 | Remote-sensing VLM adaptation: real LoRA training on BigEarthNet.txt, adapter served through the registry | Done |
| 6 | Bi-temporal change: co-registration, change vector analysis, change description + change-VQA, change map | Done |
| 7 | Optical-SAR cross-modal: feature-level fusion, complementarity, water agreement, cross-modal VQA | Done |
| 8 | Production UX, downloadable reports, accessibility audit | Next |
| 9–10 | Benchmark evaluation + calibration, Docker build/deploy, final demo | Planned |

Working today: upload optical/SAR/temporal rasters and get real validated metadata (CRS,
transform, bounds, resolution, bands, dtype); view a server-rendered preview with zoom/pan;
ask a question in plain language and watch the agent classify the task, validate the inputs,
select a registered tool, run **real model inference**, and return an answer with evidence, a
derived confidence level, and a full execution trace.

Single-image capabilities run on CPU with no GPU and no API keys:

- **VQA** — "What land cover dominates?", "Is water present?" answered from measured
  concept probabilities.
- **Captioning** — a remote-sensing scene description composed from real scores.
- **Grounding** — "Highlight the water" returns bounding boxes or a coarse region, a
  rendered overlay image, and real latitude/longitude for georeferenced rasters.

### Remote-sensing adaptation (Phase 5) — real training, real measurement

`clip-rsicd-v2` was LoRA-adapted on **8,362 real BigEarthNet.txt image–caption pairs** with
the symmetric InfoNCE objective: 1,966,080 of 153,243,393 parameters trainable (1.283 %),
6 epochs, **7.6 minutes on a verified NVIDIA RTX 2050 (4 GB)**, peak 2,219.2 MB VRAM. The
12.3 MB adapter is on disk and is served by the API as separate registry entries
(`vqa_adapted`, `captioning_adapted`), so every trace states which model actually ran.

Baseline and adapted were both evaluated on the dataset's **held-out** `test` split (n=2,053,
never used for training or validation), same preprocessing for both:

| | Baseline | Adapted |
|---|---|---|
| mean recall | 0.0115 | **0.0791** (6.9×) |
| median rank image→text | 518 | **85** |
| median rank text→image | 645 | **84** |

All six recall metrics improved. Chance R@1 for this pool is 0.00049. Details and caveats:
`docs/training.md`, `docs/evaluation.md`, `docs/datasets.md`.

**Honest scope note.** The VQA/captioning answer strategy remains zero-shot concept scoring —
adaptation improved the embedding space, it did not add a trained task head. All model entries
stay registered as `degraded`, not `available`: real inference, uncalibrated scores, and **no
benchmarked task accuracy**. Retrieval improving is not the same claim as "VQA accuracy
improved", and that claim is not made anywhere here. Grounding was **not** adapted, so no
improvement is claimed for it. Every result states its limitations. Counting questions are
**refused** rather than answered with an invented number; captions are labelled
retrieval-composed rather than generated; a failed localization is reported as inconclusive,
never as absence. Temporal change and optical-SAR fusion are routed correctly but return
`unavailable` until Phases 6–7. **No placeholder or fabricated answers are ever shown.**
See `docs/models.md` for provenance, measured latency, and limitations.

### Bi-temporal change (Phase 6) — real measurement, honest wording

Upload two dates of the same area and ask what changed. The agent validates the pair,
co-registers T1 and T2 onto a common grid (reprojecting and resampling as needed, recorded in
the trace), and runs **change vector analysis** — radiometric normalisation, robust
thresholding, and spatial-coherence filtering — to measure where and how much changed. It
produces a **change map**, a T1/T2/change comparison strip, per-region geometry with real
coordinates, and NDVI/NDWI shifts when the rasters declare their bands. Both mandatory
capabilities ship: change **description** and change **VQA**.

Change detection is **algorithmic and CPU-only** (NumPy/SciPy/Rasterio) — no model weights, no
GPU, runs anywhere. The registry lists it as `degraded`: real and reproducible, but not yet
benchmarked against a labelled change dataset (Phase 9).

**Honest scope note.** Cause of change is never asserted — there is no built-up classifier, and
"is there construction?" says so explicitly while reporting the measured change. Optical-vs-SAR
differencing is refused (fusion is Phase 7). A misregistered pair returns "Unable to determine"
rather than reporting displacement as change. NDVI/NDWI are refused unless the bands are
declared, because guessing band order would fabricate an index. Questions outside the measured
signals get a stated limitation, not a guess.

### Optical + SAR cross-modal (Phase 7) — genuine fusion, not caption concatenation

Upload a co-located optical and SAR image and ask what they show together. The agent
co-registers them onto one grid and measures quantities **neither modality produces alone**:
optical↔SAR correlation/complementarity, the optical-only / SAR-only / shared information
partition, optical-vs-SAR **water agreement** (NDWI against SAR-dark surfaces), and **SAR-only
structure regions** (what SAR reveals that optical does not) with coordinates. It answers
cross-modal description and VQA ("what does SAR reveal that optical does not?", "which modality
shows vegetation more strongly?", "are optical and SAR consistent on water?") from those
measurements, with a false-colour composite and a complementary map as evidence.

Like change detection, this is **algorithmic and CPU-only** — no weights, no GPU. SAR is treated
as **log-scaled, uncalibrated backscatter** unless the metadata declares calibration (sigma0/
gamma0 is never assumed), and SAR evidence is reported as measured backscatter/roughness, never
as land-cover classes. Two-optical or two-SAR pairs are refused; a missing modality is refused.
Registered `degraded`: real and reproducible, not yet benchmarked. Details: `docs/phase7.md`.

Tests: **404 backend, 96 frontend**. Serving is **CPU by default** and needs no GPU; Phase 5
training required a real CUDA GPU and is kept fully separable from serving.

## Capabilities (target)

- Single-image **VQA** + **grounding/captioning**
- **Bi-temporal change** understanding (description / change-VQA)
- Genuine **optical-SAR fusion**
- **Remote-sensing VLM adaptation** (LoRA/QLoRA; BigEarthNet route)
- **Agentic orchestration** with a model/tool registry, evidence, confidence, and trace

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite, Tailwind CSS, shadcn/ui, React Router, TanStack Query, Vitest, Playwright |
| Backend | Python 3.11, FastAPI, Pydantic, SQLAlchemy, Alembic, Uvicorn |
| Database | PostgreSQL (SQLite for local dev/tests) |
| Geospatial | Rasterio, GDAL, NumPy, OpenCV, scikit-image, pyproj, Shapely, GeoPandas |
| ML | PyTorch, Transformers, PEFT (LoRA/QLoRA), Accelerate, safetensors |
| Packaging | Docker, Docker Compose |

## Repository layout

```text
.kiro/          Kiro steering rules, specs, hooks
backend/        FastAPI application + tests
frontend/       React + TypeScript web app
geospatial/     Raster IO / validation / alignment (Phase 2)
ml/             Datasets, training (LoRA), inference backends, evaluation (Phases 4-5)
models/         Model checkpoints / LoRA adapters (gitignored, reproducible)
data/           Datasets (gitignored) + data/manifests/ (tracked, SHA256-pinned)
reports/        Measured training & evaluation metrics (phase5/*.json tracked)
storage/        Runtime uploads / artifacts / evidence / reports / temp (gitignored)
docs/           Architecture, setup, traceability, dataset/model/training/eval provenance
docker/         Dockerfiles
scripts/        Verification, dataset, and training scripts (each writes a report file)
```

## Quick start (local, CPU)

Prerequisites: Python 3.11+, Node.js 20+, Docker Desktop, Git.

```powershell
# 1. Configure environment (optional: defaults work with no .env)
Copy-Item .env.example .env

# 2. Backend
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -e .                # shared geospatial library
pip install -e "backend[dev]"   # API + dev tools
alembic -c backend/alembic.ini upgrade head
uvicorn app.main:app --app-dir backend --reload --port 8000
# API docs: http://localhost:8000/docs

# 3. Frontend (separate terminal)
cd frontend
npm install
npm run dev
# UI: http://localhost:5173
```

## Run with Docker (clean machine)

```powershell
Copy-Item .env.example .env   # set POSTGRES_PASSWORD
docker compose up --build
# Frontend: http://localhost:8080   API: http://localhost:8000/docs
```

> Tip: this machine's C: drive is low on space. Point Docker Desktop's disk image
> and any model/dataset downloads at the D: drive.

The image installs `peft`, so it can serve the adapted model. The adapter is **not** baked
into the image — copy `models/phase5/adapter/` into the `models` volume to serve it. Without
it the backend serves the Phase 4 baseline and records the fallback in the execution trace.

## Reproduce the Phase 5 adaptation (needs a CUDA GPU)

```powershell
$env:HF_HOME="D:\SatQuery-AI\models\hf-cache"
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[training]"

.\.venv\Scripts\python.exe scripts\verify_phase5_gpu.py        # hard gate: real GPU
.\.venv\Scripts\python.exe scripts\download_bigearthnet.py     # ~2.75 GB
.\.venv\Scripts\python.exe scripts\prepare_bigearthnet_txt.py  # splits + SHA256 manifest
.\.venv\Scripts\python.exe scripts\phase5_smoke_train.py       # 13-stage pipeline check
.\.venv\Scripts\python.exe scripts\train_phase5.py --epochs 6 --batch-size 32
.\.venv\Scripts\python.exe scripts\evaluate_phase5.py --split test
.\.venv\Scripts\python.exe scripts\verify_phase5.py            # 25 end-to-end checks
```

Training refuses to run on CPU rather than silently producing a misrepresented result. Each
script writes a report file next to itself, which is the reliable record (this shell truncates
long-running stdout).

## Documentation

- `SatQuery_AI_Kiro_Master_Plan.md` — master engineering specification (source of truth)
- `docs/architecture.md` — system architecture
- `docs/setup.md` — detailed setup & troubleshooting
- `docs/api.md` — API reference
- `docs/models.md` — model provenance, licensing, measured latency, and limitations
- `docs/datasets.md` — dataset provenance, licensing, splits, exclusions, preprocessing
- `docs/training.md` — Phase 5 adaptation: method, hyperparameters, measured results, repro
- `docs/phase6.md` — Phase 6 bi-temporal change: method, decisions, honesty mechanisms, repro
- `docs/phase7.md` — Phase 7 optical+SAR cross-modal fusion: method, why genuine, repro
- `docs/evaluation.md` — what has been measured, and what is `NOT EVALUATED` / `NOT BENCHMARKED`
- `docs/limitations.md` — honest, capability-by-capability limitations
- `docs/final_completion_ledger.md` — project-wide remaining-work tracker (Phases 8–10)
- `docs/traceability.md` — SIH requirement → implementation → test → UI → demo matrix
- `.kiro/specs/` — per-feature requirements, design, and tasks

## License

MIT — see `LICENSE`. (Default choice; change if your submission requires otherwise.)
