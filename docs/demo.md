# Demo Walkthrough — SatQuery AI (SIH26167)

A complete judge/demo guide, runnable from a clean checkout. Every step uses the **real**
backend: the demo imagery is synthetic (generated locally, no hidden evaluation data), and the
system genuinely analyzes it — nothing in the UI is hard-coded.

> **Benchmark caveat (read first):** all model/tool outputs are real but **not benchmark-
> validated or calibrated**. Confidence is derived from measurable signals, not a calibrated
> probability. Treat every result as *indicative*. See `docs/evaluation.md` and
> `docs/limitations.md`.

---

## 1. Prerequisites
- Python 3.11+, Node.js 20+, Git. (Docker optional — see §6.)
- ~1.2 GB free for the Phase 4/5 model weights (downloaded on first single-image analysis).
- No GPU required. Everything in this demo runs on CPU.

## 2. Installation
```powershell
# From the repository root
python -m venv .venv
./.venv/Scripts/Activate.ps1          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -e .                       # shared geospatial + ML libraries (satquery-core)
pip install -e "backend[dev]"          # API + dev tools
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU torch for inference
pip install "transformers>=4.44,<5" "safetensors>=0.4" peft         # single-image models
cd frontend && npm install && cd ..
```

## 3. Environment setup
No `.env` is required — defaults use a local SQLite database and cache weights under
`models/hf-cache`. To pin the cache location (recommended on this machine, D: has space):
```powershell
$env:HF_HOME = "D:\SatQuery-AI\models\hf-cache"
$env:DEVICE  = "cpu"
```

## 4. Start the backend
```powershell
.\.venv\Scripts\python.exe -m alembic -c backend/alembic.ini upgrade head   # once
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000
```
API docs: `http://localhost:8000/docs` · health: `http://localhost:8000/api/v1/health`.

## 5. Start the frontend
```powershell
cd frontend
npm run dev
```
Open `http://localhost:5173`.

## 6. Docker option
```powershell
Copy-Item .env.example .env    # set POSTGRES_PASSWORD
docker compose up --build      # frontend :8080, API :8000
```
> Note: Docker image **build** on this machine is disk-constrained; `docker compose config` is
> validated. See `docs/final_completion_ledger.md` (DEP-2/DEP-3) for status.

## 7. Load the demo data
```powershell
.\.venv\Scripts\python.exe scripts\make_demo_data.py
```
This writes synthetic, deterministic rasters to `data/demo/` and a checksummed manifest to
`data/manifests/demo_manifest.json`. Files:

| File | Use |
|---|---|
| `demo_optical.tif` | single optical (3-band: Red, Green, NIR) |
| `demo_sar.tif` | single SAR-like backscatter (1-band) |
| `demo_optical_t1.tif`, `demo_optical_t2.tif` | temporal pair (T2 contains a change block) |
| `demo_incompatible.tif` | optical with a non-overlapping footprint (for the invalid demo) |

Source: **synthetic**, CC0, no attribution required, no hidden data. Regenerate any time — the
output is byte-identical (fixed seeds), so the manifest checksums are stable.

## 8. Single-image VQA
1. In **Inputs**, upload `demo_optical.tif` into the **Optical / Multispectral** slot.
2. In the **Assistant**, click the chip *"What land cover dominates this image?"* (or type it).
3. Click **Analyze**.
4. You should see: task **Visual Question Answering**, a **Completed** answer with measured
   concept probabilities, a **confidence** level with its signals, and **evidence** (score bars,
   model provenance, preview).

## 9. Caption / grounding
- **Caption:** with `demo_optical.tif` loaded, ask *"Describe this scene"* → task **Captioning**,
  a retrieval-composed description (labelled as such).
- **Grounding:** ask *"Highlight the water body in this image"* → task **Grounding**, a rendered
  overlay (bounding boxes) or a labelled coarse region, with lon/lat when georeferenced.

## 10. Adapted VLM (Phase 5)
With the Phase 5 adapter present (`models/phase5/adapter/`), a VQA/captioning query on optical
imagery automatically routes to the **adapted** model. Look for the **"Adapted model"** badge
and the selection note *"selected the Phase 5 adapted model (adapter present)"* in the result,
and the model provenance (base model + adapter + training dataset) in the execution trace. If
the adapter is absent, the baseline runs and the trace says so — never a silent swap.

## 11. Temporal change
1. Upload `demo_optical_t1.tif` → **Temporal T1**, `demo_optical_t2.tif` → **Temporal T2**.
2. (Optional) Click **Check T1 / T2** to see the compatibility verdict.
3. Ask *"What changed between these two dates?"*.
4. Expect task **Temporal Change** / **Change VQA**, a grounded change description (measured
   extent, location, brightness direction), and a **change map** plus a **T1/T2/change** strip in
   the Visualization panel's *Change evidence* view.

## 12. Optical + SAR cross-modal
1. Upload `demo_optical.tif` → **Optical**, `demo_sar.tif` → **SAR**.
2. Ask *"What complementary information does SAR provide?"*.
3. Expect task **Optical-SAR Fusion**, a measured cross-modal answer (correlation/
   complementarity, SAR-only structure, water agreement), and the **Optical + SAR** view with a
   false-colour composite and a complementary map.

## 13. Invalid-input demonstration
1. Upload `demo_optical_t1.tif` → **T1**, `demo_incompatible.tif` → **T2**.
2. Ask *"What changed between these two dates?"*.
3. Expect a **Rejected** result explaining that the images do not overlap geographically — and
   **no fabricated change answer**. This shows the anti-fabrication guard.

## 14. Interpreting evidence
Each evidence card names its kind (metadata, preview, overlay, change map, fusion composite,
complementary region, modality comparison, water agreement, scores, model provenance) and shows
the measured values behind it. Images have descriptive alt text; if an image cannot load, a
labelled fallback is shown instead of a broken icon.

## 15. Interpreting the trace
The **Execution trace** is an ordered, observable log: input inspection → query understanding →
task classification → validation → tool selection → model provenance → co-registration (pairs) →
execution → evidence → confidence. It contains **no chain-of-thought** — only what the system
did and each step's outcome.

## 16. Interpreting confidence
Levels are **High / Medium / Low / Unable to determine**, each derived from listed signals
(result strength, routing clarity, input georeferencing, evidence, warnings). "Unable to
determine" is a real, honest outcome — used for inconclusive or unanswerable cases — not a
failure to try.

## 17. Downloadable report
On any result, click **Download report**. This fetches a self-contained HTML report
(`GET /api/v1/analysis/{id}/report`) with the query, task/tool, answer, confidence + signals,
embedded evidence images, the trace, provenance, and limitations. Open it in a browser and use
**Print → Save as PDF** for a PDF copy.

## 18. Troubleshooting
See `docs/troubleshooting.md`. Common: first single-image analysis is slow (one-time model
load); if it returns *unavailable* with `model_unavailable`, the weights are not cached — run a
single-image query once while online, or pre-cache with `python scripts/check_models.py`.

## 19. Limitations
See `docs/limitations.md`. In short: nothing is benchmarked; scores are uncalibrated; SAR is
uncalibrated backscatter; change detection is algorithmic (not a learned CD model); the adapted
VLM was trained on one country/season.

## 20. Benchmark caveat (restated)
No RSVQA / VRSBench / CDVQA / change / fusion benchmark has been run. Do not present any number
here as benchmark accuracy. Benchmarking is Phase 9.

## Reset the demo
Delete the local dev database and evidence to start clean:
```powershell
Remove-Item storage\satquery_dev.db -ErrorAction SilentlyContinue
Remove-Item -Recurse storage\evidence -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m alembic -c backend/alembic.ini upgrade head
```
