# SatQuery AI

**An Interactive Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis through Text Queries**

- **Problem statement:** SIH26167 — SatQuery AI
- **Organization:** ISRO / Department of Space
- **Category:** Software · **Theme:** Space Technology

SatQuery AI is an **agentic** remote-sensing analysis assistant. A user uploads satellite imagery (optical, SAR, a bi-temporal pair, or co-registered optical + SAR), asks a question in plain language, and the system understands the query, validates the inputs, selects the right specialist model/tool, runs the analysis, and returns an **answer with visual evidence, a confidence level, and an auditable execution trace**.

It is a real end-to-end system, not a chatbot or a UI mock — every mandatory capability is implemented, wired into the API and UI, and covered by tests. Nothing is answered with a fabricated or placeholder result.

---

## Key capabilities

- **Single-image VQA** — land-cover / presence questions answered from measured concept probabilities.
- **Captioning** — generative scene descriptions (trained BLIP) with a CLIP retrieval-based captioner as fallback.
- **Grounding** — "highlight the water" returns bounding boxes or a coarse region, a rendered overlay, and real lat/long for georeferenced rasters.
- **Bi-temporal change** — co-registration + change-vector analysis producing a change map, T1/T2 comparison, per-region geometry, and NDVI/NDWI shifts; plus change-VQA.
- **Optical + SAR fusion** — genuine feature-level cross-modal analysis (correlation/complementarity, water agreement, SAR-only structure), not text concatenation.
- **Remote-sensing VLM adaptation** — a real LoRA adapter trained on BigEarthNet.txt, served through the registry.
- **Agentic orchestration** — registry-constrained tool/model selection with evidence, confidence, and a full execution trace.

Serving runs on **CPU by default** — no GPU and no API keys required. Model training is kept fully separable from serving.

## Implemented models

The agent selects only from a validated registry. Each result names the exact model/tool that produced it.

| Route | Model / method | Type | Default |
|---|---|---|---|
| Captioning | `captioning_blip` — BLIP (`BlipForConditionalGeneration`), VRSBench fine-tune | Trained, generative | ✅ (CLIP captioner as fallback) |
| Change-VQA | `change_vqa_blip` — BLIP (`BlipForQuestionAnswering`), CDVQA + SECOND fine-tune | Trained, generative | ✅ (deterministic route as fallback) |
| Change-VQA | `cdvqa_trained`, `cdvqa_transfer` — CLIP-feature classifiers | Trained, closed-vocabulary | Selectable |
| VQA / Captioning | `clip-rsicd-v2` (+ Phase 5 LoRA adapter) | Zero-shot / adapted | Baseline |
| Grounding | `owlvit-base-patch32` | Pretrained (open-vocabulary) | ✅ |
| Temporal change / Change-VQA | Change-vector analysis | Deterministic (no weights) | Fallback |
| Optical + SAR fusion | Feature-level cross-modal analysis | Deterministic (no weights) | ✅ |

Model selection is configurable via `CHANGE_VQA_MODEL` and `CAPTIONING_MODEL`; if a trained checkpoint is unavailable, routing falls back automatically so functionality never breaks. All learned models are registered `degraded` (real inference, uncalibrated scores) rather than `available`.

## Benchmarked results

Measured on held-out splits with a documented protocol (train/val/frozen-test separation, no test-set tuning). See `docs/evaluation.md`.

| Model | Benchmark | Result |
|---|---|---|
| Phase 5 LoRA adaptation | BigEarthNet.txt retrieval (held-out, n=2,053) | mean recall **0.0115 → 0.0791** (≈6.9×) |
| `cdvqa_trained` | CDVQA held-out | Test **0.7114** / Test2 **0.6632** |
| `cdvqa_transfer` (CLIP + BLIP features) | CDVQA held-out | Test **0.7156** / Test2 **0.6668** |
| `change_vqa_blip` (BLIP) | CDVQA held-out (n=600) | Test **0.5467** / Test2 **0.5033** |

**Unbenchmarked:** `captioning_blip` has only training-loss curves — no caption metric (BLEU/CIDEr) has been measured, so no captioning accuracy is claimed. The single-image VQA/captioning strategy is concept scoring / generation, not a benchmarked task head. Deterministic change and fusion are reproducible but not yet scored against a labelled benchmark.

## Agentic workflow

```
User query + imagery
  → query understanding (intent classification, prompt-injection sanitization)
  → input validation (raster / CRS / pair-compatibility)
  → tool & model selection (registry allowlist; config-driven with fallback)
  → specialist execution (real inference)
  → visual evidence  +  confidence (High / Medium / Low / Unable)
  → answer  +  auditable execution trace
```

The trace records observable steps only — never chain-of-thought. Incompatible pairs are rejected, unsupported queries return "Unable to determine", and counting questions are refused rather than answered with an invented number.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite, Tailwind CSS, shadcn/ui, TanStack Query, framer-motion, three / react-three-fiber, Vitest, Playwright |
| Backend | Python 3.11, FastAPI, Pydantic, SQLAlchemy, Alembic, Uvicorn |
| Database | PostgreSQL (SQLite for local dev / tests) |
| Geospatial | Rasterio, GDAL, NumPy, SciPy, pyproj |
| ML | PyTorch, Transformers, PEFT (LoRA), safetensors |
| Packaging | Docker, Docker Compose |

## Repository layout

```text
backend/     FastAPI application + tests
frontend/    React + TypeScript web app
geospatial/  Raster IO / validation / alignment
ml/          Datasets, training, inference backends, evaluation
models/      Model checkpoints / LoRA adapters (gitignored, reproducible)
data/        Datasets (gitignored) + data/manifests/ (tracked, SHA256-pinned)
reports/     Measured training & evaluation metrics
docs/        Architecture, setup, API, model/dataset/training/eval provenance
docker/      Dockerfiles
scripts/     Verification, dataset, and training scripts
```

## Quick start (local, CPU)

Prerequisites: Python 3.11+, Node.js 20+, Git.

```powershell
# 1. (optional) configure environment — defaults work with no .env
Copy-Item .env.example .env

# 2. Backend
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -e .                # shared geospatial library
pip install -e "backend[dev]"   # API + dev tools
alembic -c backend/alembic.ini upgrade head
uvicorn app.main:app --app-dir backend --reload --port 8000   # http://localhost:8000/docs

# 3. Frontend (separate terminal)
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

**One-command run (Windows):** `./run-local.ps1` starts the backend and frontend if they are not already running, waits until both are ready, and opens the app.

## Run with Docker

```powershell
Copy-Item .env.example .env     # set POSTGRES_PASSWORD
docker compose up --build
# Frontend: http://localhost:8080   API: http://localhost:8000/docs
```

Model weights are cached on a volume (not baked into the image) and download on first use. Trained checkpoints and adapters are placed under `models/` / the checkpoint folders; if absent, the backend serves the fallback route and records it in the trace.

## Testing

```powershell
# Backend
.\.venv\Scripts\python.exe -m pytest backend/tests -q
.\.venv\Scripts\python.exe -m ruff check backend/app backend/tests geospatial ml scripts

# Frontend (in ./frontend)
npm run typecheck && npm run lint && npm run test && npm run build
npx playwright test          # end-to-end browser flows (real backend)
```

Current suite: **468 backend tests** and **126 frontend tests** passing, plus 6 Playwright end-to-end flows. Tests run without a GPU and without a running PostgreSQL (SQLite).

## Documentation

| Doc | Contents |
|---|---|
| `SatQuery_AI_Kiro_Master_Plan.md` | Master engineering specification |
| `docs/architecture.md` | System architecture |
| `docs/setup.md` | Detailed setup & troubleshooting |
| `docs/api.md` | API reference |
| `docs/models.md` | Model provenance, licensing, latency, limitations |
| `docs/datasets.md` | Dataset provenance, licensing, splits, preprocessing |
| `docs/training.md` | Phase 5 adaptation: method, hyperparameters, results |
| `docs/evaluation.md` | What is measured vs. `NOT BENCHMARKED` |
| `docs/limitations.md` | Honest, capability-by-capability limitations |
| `docs/traceability.md` | SIH requirement → implementation → test → UI matrix |

## Datasets

Only public/open datasets are used, with documented provenance and licensing; **no hidden ISRO/SAC evaluation data** is used anywhere.

- **RSICD** — base for `clip-rsicd-v2`.
- **BigEarthNet.txt** — Phase 5 LoRA adaptation.
- **CDVQA + SECOND** — change-VQA training and evaluation.
- **VRSBench** — BLIP captioning fine-tune.
- **LEVIR-CD / OSCD** — cross-dataset change-detection evaluation.
- Synthetic demo rasters (`data/manifests/`, SHA256-pinned) for reproducible demos.

## Limitations (honest)

- Learned models are `degraded`: real inference with **uncalibrated** scores; confidence signals are indicative, not calibrated probabilities.
- `captioning_blip` is **not benchmarked**; `change_vqa_blip` scores below `cdvqa_trained` on CDVQA and is a single-image model (a bi-temporal pair is shown under a fixed convention).
- CLIP-based VQA/captioning has no counting or spatial-reasoning head; counting is refused.
- Change detection reports *that*/where/how-much change occurred, not its semantic cause.
- SAR is treated as uncalibrated log-backscatter unless metadata declares calibration.
- Models are trained on specific sensors/regions; generalization to other sensors/resolutions is not established.
- API endpoints are unauthenticated (acceptable for the local demo; authentication is required before any public deployment).

## License

MIT — see `LICENSE`.
