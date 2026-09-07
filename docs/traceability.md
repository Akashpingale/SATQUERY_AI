# Requirement Traceability Matrix — SatQuery AI (SIH26167)

Maps every mandatory/required SIH requirement to its implementation, module, test, UI
surface, demo, and evidence. **Status is honest**: it reflects what is actually built,
not what is planned. Updated as each phase lands.

Legend — Status: `Done` · `In progress` · `Planned`. Phase = master-plan phase.

| # | SIH Requirement | Priority | Phase | Implementation | Code / Module | Test | UI | Demo | Evidence | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | Natural-language queries | MANDATORY | 3 | Query understanding + classification | `backend/app/agent/understanding.py` (04) | `test_query_understanding.py` (routing, shape, injection) | Assistant query box | 1–6 | Selected task in trace | **Done** |
| R2 | Interactive GUI / web app | MANDATORY | 1–8 | React app | `frontend/src/` (01,02) | RTL + Playwright | Whole app | 1–6 | Screens | In progress |
| R3 | Optical/multispectral support | MANDATORY | 2 | Raster ingestion | `geospatial/`, `backend/app/services/ingestion.py` (03) | `test_raster_metadata.py`, `test_uploads_api.py` | Upload slot + viewer | 1,2,5 | Metadata panel | **Done** |
| R4 | SAR support | MANDATORY | 2,7 | SAR ingest (Phase 2) + feature-level optical–SAR fusion (Phase 7) | `geospatial/`, `ml/inference/fusion.py` (03,11) | SAR ingest tests, `test_cross_modal.py` | SAR upload slot | 5 | SAR metadata/preview | **Done** (fusion `degraded`: not benchmarked) |
| R5 | GeoTIFF/TIFF | MANDATORY | 2 | Rasterio readers | `geospatial/io.py`, `metadata.py` (03) | format + fixture tests | Format hints, metadata | 1–6 | CRS/bounds/bands | **Done** |
| R6 | Input/metadata validation | MANDATORY | 2 | Validators + pair checks | `geospatial/validation.py`, `alignment.py`, `backend/app/security/files.py` (03,15) | negative tests (corrupt/exe/empty/traversal/no-overlap) | Validation status + issue list | 3,4,5 | Warnings/errors | **Done** |
| R7 | Single-image VQA | MANDATORY | 4,5 | VQA on RS-adapted CLIP; since Phase 5 it runs on the **LoRA-adapted** model when the adapter is present, baseline otherwise | `ml/inference/clip_backend.py`, `backend/app/agent/specialists/vqa.py` (08) | `test_vision_specialists.py`, `test_phase5_adaptation.py` | Answer card + score bars + which model ran | 1 | Answer + measured probability + model provenance | **Implemented** (`degraded`: uncalibrated, unbenchmarked) |
| R8 | Additional single-image task | MANDATORY | 4,5 | **Both** captioning and grounding. Captioning uses the adapted model; grounding deliberately unchanged (not adapted, so no gain claimed) | `.../specialists/{captioning,grounding}.py`, `ml/inference/{detection_backend,overlay}.py` (09) | rendering, coords, all 3 grounding outcomes; adapted-preference test | Overlay + geometry + coords | 2 | Boxes/regions + lon/lat | **Implemented** (`degraded`) |
| R9 | RS VLM fine-tuning/adaptation | MANDATORY | 5 | **Real LoRA adaptation of `clip-rsicd-v2`** on BigEarthNet.txt (InfoNCE, 1.97 M trainable params, RTX 2050); adapter served via registry as `vqa_adapted`/`captioning_adapted` | `ml/training/lora_clip.py`, `ml/inference/adapted_clip.py`, `ml/evaluation/retrieval.py` (07) | `test_phase5_adaptation.py`; `scripts/verify_phase5.py` (25/25) | Adapted model + provenance shown in trace and evidence | 1,2 | `reports/phase5/*.json`; held-out baseline-vs-adapted retrieval (mean recall 0.0115 → 0.0791) | **Done** (`degraded`: one country/season, uncalibrated) |
| R10 | BigEarthNet / OSS RS data route | MANDATORY | 5 | BigEarthNet.txt pipeline: official annotations, official splits, recorded exclusions, SHA256-pinned manifest | `ml/datasets/bigearthnet.py`, `scripts/{download,prepare}_bigearthnet*.py` (07) | `test_phase5_adaptation.py` (manifest/report contract) | — | — | `data/manifests/bigearthnet_txt_manifest.json`; 8,362 retained / 0 excluded; `docs/datasets.md` | **Done** (imagery via an **unofficial** mirror — validated against official annotations, 0 mismatches) |
| R11 | Bi-temporal analysis | MANDATORY | 6 | Safe co-registration to a common grid (WarpedVRT) + change vector analysis; misregistration refused | `geospatial/resample.py`, `ml/inference/temporal.py` (10) | `test_change_detection.py`, `test_temporal_api.py` (known-truth + negatives) | T1/T2 compare + change map views | 3,4 | Change map, T1/T2/change strip, per-region geometry | **Done** (`degraded`: not benchmarked) |
| R12 | Change description OR change-VQA | MANDATORY | 6,9 | **Both**: grounded change description + measurement-based change VQA (Phase 6); **plus a Phase 9 learned CDVQA classifier trained on real CDVQA data** (held-out Test 0.71/Test2 0.66, calibrated) registered as `cdvqa_trained` | `backend/app/agent/specialists/temporal.py`, `cdvqa_trained.py`; `scripts/train_cd_vqa.py` | classification/honesty tests; `test_cdvqa.py` | Answer + confidence + evidence | 3,4 | Measured extent/indices; `reports/benchmarks/cdvqa/`, `docs/CDVQA_TRAINING_REPORT.md` | **Done** — deterministic default `degraded`; learned CDVQA benchmarked (`degraded`, non-default) |
| R13 | Optical + SAR joint analysis | MANDATORY | 7 | **Feature-level cross-modal fusion**: co-register (align_modalities) → per-modality features → correlation/complementarity + information partition + optical/SAR water agreement + SAR-only structure regions. Not caption concatenation | `ml/inference/fusion.py`, `backend/app/agent/specialists/fusion.py` (11) | `test_cross_modal.py`, `test_fusion_api.py` (known-truth + negatives) | Optical+SAR view: composite, complementary map, aligned previews, modality comparison | 5 | Fusion composite + complementary regions + water agreement | **Done** (`degraded`: not benchmarked) |
| R14 | Agentic orchestration | MANDATORY | 3 | Full pipeline orchestrator | `backend/app/agent/orchestrator.py` (05) | `test_analysis_api.py` E2E | Assistant panel | 6 | Execution trace | **Done** |
| R15 | Automatic task/model/tool selection | MANDATORY | 3 | Registry-constrained planner | `backend/app/agent/planner.py`, `registry/` (05,06) | `test_agent_planning.py` | Selected tool + version shown | 6 | Trace | **Done** (per-tool availability varies) |
| R16 | Workflow sequencing | MANDATORY | 3 | Multi-step plans; validate-first on pairs | `backend/app/agent/planner.py` (05) | toolchain order tests | Trace steps | 3,4,5 | Trace | **Done** |
| R17 | Controlled parameters | REQUIRED | 3 | Allowlisted `ParamSpec` validation | `backend/app/agent/validation.py`, `registry/models.py` (06) | param allowlist tests | — | 6 | Safe params summary | **Done** |
| R18 | Text/spatial output combination | REQUIRED | 4–7 | Answer + boxes/regions + lon/lat in one envelope | `.../specialists/grounding.py`, `geospatial/coords.py` (12) | coordinate + evidence tests | Answer + overlay + coords | 1–5 | Linked spatial evidence | **Done** for single image; pairs in Phase 6/7 |
| R19 | Visual evidence | REQUIRED | 4+ | Rendered overlays, boxes, regions, score bars, previews | `ml/inference/overlay.py`, `.../specialists/` (12) | rendering + endpoint tests | Evidence panel (overlay-first) | 1–6 | PNG overlays served via API | **Done** (masks still pending) |
| R20 | Confidence | REQUIRED | 4+ | Signal-derived assessment | `backend/app/agent/confidence.py` (12) | `test_confidence_trace.py` | Confidence badge + signals | 1–6 | Level + listed signals | **Done** |
| R21 | Observable execution summary | REQUIRED | 3+ | Trace builder, no chain-of-thought | `backend/app/agent/trace.py` (12) | trace safety tests | Trace panel | 1–6 | Ordered steps | **Done** |
| R22 | Benchmark evaluation | REQUIRED | 9 | Phase 9 eval framework routing through production specialists; **real held-out training/eval executed**: CDVQA (Test 0.71/Test2 0.66, calibrated ECE 0.014), SECOND learned change (IoU 0.42 vs 0.22), BigEarthNet retrieval (6.9×). Synthetic suites also executed. RSVQA/VRSBench/LEVIR-CD/OSCD **NOT RUN** (unprovisioned; VQA/caption architecture-incompatible) | `ml/evaluation/*`, `scripts/{train_cd_vqa,calibrate_cdvqa,train_second_change,run_benchmarks,eval_external}.py` | `test_evaluation.py`, `test_cdvqa.py`, `test_second.py` | — | — | `reports/benchmarks/*`; `docs/{CDVQA_TRAINING_REPORT,SECOND_TRAINING_REPORT,benchmarking,benchmark_report}.md` | **Partial→strong** — two real trained benchmarks + retrieval measured; RSVQA/VRSBench NOT RUN, no fabrication |
| R23 | Hidden ISRO/SAC readiness | CRITICAL | all | No hacks + config | across (17) | robustness tests | — | — | Generalization notes | Planned |
| R24 | Change map | OPTIONAL | 6 | Magnitude-ramped change map + T1/T2/change strip, rendered from the measured mask | `ml/inference/change_map.py` (10) | render tests (tint only on measured pixels) | Change map view + evidence panel | 3 | PNG served via evidence token | **Done** |
| R25 | Downloadable report | HIGH-VALUE | 8 | Self-contained HTML report built from the persisted record; evidence PNGs embedded as base64 data URIs; XSS-safe escaping; served via API | `backend/app/reports/html_report.py`, `GET /api/v1/analysis/{id}/report` | `test_report.py` (12: escaping, embedding, provenance, from-history) | "Download report" link in `ResultCard.tsx` | 1–6 | Portable `.html` file (print-to-PDF) | **Done** |
| R26 | Docker clean-machine run | HIGH-VALUE | 1,10 | Compose | `docker/`, `docker-compose.yml` (16) | compose smoke | — | — | Clean-run log | In progress |
| R27 | Automated tests | ENG QUALITY | all | pytest (SQLite, no GPU) + vitest + Playwright E2E on the real workflow | `backend/tests/`, `frontend/tests/`, `frontend/e2e/flows.spec.ts` | **416 backend + 119 frontend + 6/6 E2E**, 0 failures | — | 1–6 | Verified run (Phase 8 report) | **Done** (CI wiring pending) |
| R28 | Accessibility | ENG QUALITY | 2–8 | Semantic HTML, keyboard nav, labels, status-not-by-colour; automated axe checks | `frontend/`, `frontend/tests/accessibility.test.tsx` | axe: 13 tests, 0 violations (only `color-contrast` disabled — jsdom limitation, documented); **manual NVDA NOT YET VERIFIED** | Whole app | — | `docs/accessibility.md` | **Partial** — axe/keyboard done; NVDA not yet run |
| R29 | Security (uploads/agent/secrets) | ENG QUALITY | all | Upload guards (MIME/size/corruption/path confinement/safe names) + prompt-injection sanitization + registry-allowlisted params; API is unauthenticated (flagged) | `backend/app/security/files.py`, `backend/app/agent/understanding.py` (`sanitize_query`), `backend/app/agent/validation.py` | `test_raster_validation.py` negatives, query-injection tests | Safe errors in UI | — | `docs/security.md`; rejection/blocked-fragment logs | Upload + prompt-injection done; **auth NOT implemented** (Phase 10) |
| R30 | Database (metadata) | REQUIRED | 1 | PostgreSQL + SQLAlchemy | `backend/app/db/`, `models_db/` (13) | migration/API tests | History | — | Stored records | Schema live (analyses, raster_assets) |
| R31 | Versioned API | REQUIRED | 1 | FastAPI `/api/v1` | `backend/app/api/v1/` (13) | API tests | — | — | OpenAPI at `/docs` | health/models/tasks/uploads live |

\*Registry live = the model/tool registry is served and enforced; individual specialists
still report honest `planned` status until implemented. Phase 2 flipped
`raster_preprocessor` and `registration_validator` to `available` because they are real.

## Status summary (end of Phase 8)

**Phase 8 delivered production UX, evidence, and quality gates on top of the Phase 1–7
capabilities — verified, not planned.**

- **R25 downloadable report — Done.** `GET /api/v1/analysis/{id}/report` renders a
  self-contained HTML report from the persisted record with evidence PNGs embedded as
  base64 data URIs (XSS-safe). "Download report" link in `ResultCard.tsx`. 12 tests.
- **R27 automated tests — Done** (CI wiring still pending): **416 backend** (SQLite, no
  GPU), **119 frontend**, and **6/6 Playwright E2E** flows on the real production
  workflow with deterministic synthetic fixtures — no mocked model outputs.
- **R28 accessibility — Partial.** axe automation (13 tests, 0 violations; only
  `color-contrast` disabled due to a jsdom layout limitation, documented and checked
  manually) plus keyboard nav and semantic HTML. **Manual NVDA screen-reader pass is NOT
  YET VERIFIED** — the procedure is documented in `docs/accessibility.md` but has not
  been run, and is not claimed as complete.
- **R29 security — prompt-injection handling documented and tested** alongside the
  existing upload guards; the `/api/v1` endpoints remain **unauthenticated** and this is
  flagged explicitly (see `docs/security.md`), to be resolved in Phase 10.
- **Demo readiness:** a deterministic synthetic demo dataset (`scripts/make_demo_data.py`
  + tracked sha256 manifest, CC0) and `docs/demo.md` walkthrough were added. All six
  demo flows pass under E2E; a clean-machine deploy rehearsal remains for Phase 10.
- **New/updated docs:** `docs/api.md`, `docs/user-guide.md`, `docs/security.md`,
  `docs/troubleshooting.md`, `docs/demo.md`, `docs/accessibility.md`.
- **Regression:** ruff / eslint / tsc clean; `vite build` (JS 296 KB, gzip 90 KB) and
  `docker compose config` pass. **0 failures** across all suites.
- **Deliberately still open:** benchmarks + calibration (Phase 9), Docker image build +
  clean-machine deploy, git initial commit, API auth, dependency/secret scan, and
  LICENSE/attribution consolidation (Phase 10). None are faked.

## Status summary (end of Phase 7)

**All six mandatory remote-sensing capabilities are now implemented and tested.** R13 —
optical–SAR cross-modal analysis — is satisfied by a **genuine feature-level fusion**, not text
concatenation:

- `geospatial.align_modalities` co-registers the optical and SAR images onto one grid (reusing
  the Phase 6 grid planner), keeping all optical bands and the SAR band, recording every
  operation.
- `ml.inference.analyze_cross_modal` measures quantities neither modality produces alone:
  optical↔SAR **correlation/complementarity**, the **optical-only / SAR-only / shared**
  information partition, **optical/SAR water agreement** (NDWI vs SAR-dark, IoU-based), and
  **SAR-only structure regions** with coordinates.
- The `optical_sar_fusion` specialist answers cross-modal description and VQA (what SAR reveals,
  which modality shows a feature, agreement, complementary information) from those measurements,
  and the frontend has an Optical+SAR view with a false-colour composite, a complementary map,
  and aligned previews.

**Exactly which code path satisfies R13:** `POST /analysis` with one optical + one SAR asset →
`understand` routes to `TaskType.OPTICAL_SAR_FUSION` → planner
`["registration_validator", "optical_sar_fusion"]` → `OpticalSARFusionSpecialist.run` →
`analyze_fusion` → `align_modalities` + `analyze_cross_modal` → measured answer + evidence +
confidence + trace. Verified by `test_fusion_api.py`.

**What is NOT claimed.** Not benchmarked (Phase 9), so status is `degraded`. SAR is uncalibrated
backscatter unless metadata declares otherwise; no sigma0/gamma0 assumed. No semantic SAR
land-cover classes; built-up questions state no classifier exists. Optical-vs-SAR fusion needs
exactly one of each modality (two-optical/two-SAR refused). Coordinates and m² areas only when
georeferencing/CRS support them. The compliance is the measured cross-modal analysis, **not**
the presence of two upload boxes.

## Status summary (end of Phase 6)

**Both mandatory bi-temporal capabilities (R11/R12) are now real, plus the optional change map
(R24).** Concretely:

- **Safe co-registration exists** where it did not before: `geospatial.align_pair` reads both
  dates onto one grid via `WarpedVRT`, reprojecting a CRS mismatch, resampling to the *coarser*
  resolution so nothing is upsampled, cropping to the overlap, and recording every operation
  for the trace. A pair that cannot be aligned raises rather than comparing mismatched pixels.
- **Change detection is real and deterministic:** radiometrically normalised change vector
  analysis (SAR log-ratio), robust noise-floor thresholding, and spatial-coherence filtering.
  Measured against known synthetic ground truth — a 30×40 block is measured at exactly 1200
  pixels, an identical or illumination-only pair at zero.
- **Change description and change VQA both ship.** Answers are composed from measured extent,
  location (from labelled connected components), brightness direction, and NDVI/NDWI shifts
  where the raster declares its bands.
- **The change map and T1/T2/change strip** are rendered from the measured mask and served
  through the existing evidence-token endpoint; a test asserts tint appears only on measured
  pixels.
- Routing, registry (`temporal_change`, `change_vqa` now `degraded`, not `planned`),
  evidence-based confidence, and an auditable no-chain-of-thought trace are all wired.

**What is deliberately NOT claimed.** No change-detection or CDVQA **benchmark** was run
(Phase 9), so status stays `degraded`. Cause of change is never asserted — there is no built-up
classifier, and built-up questions say so. Optical-SAR differencing is refused (Phase 7 does
fusion). Misregistered pairs return "Unable to determine". NDVI/NDWI are refused unless bands
are declared. A change-VQA question outside the measured signals returns a limitation, not a
guess. `ml/evaluation/change_vqa.py` is a benchmark-ready interface only — its metrics report
coverage alongside accuracy, and the CDVQA schema is documented as unverified.

## Status summary (end of Phase 5)

**The mandatory remote-sensing adaptation requirement (R9/R10) is now satisfied by real
training, not by a plan.** Concretely:

- `flax-community/clip-rsicd-v2` was LoRA-adapted on **8,362 real BigEarthNet.txt
  image–caption pairs** with the symmetric InfoNCE objective — 1,966,080 of 153,243,393
  parameters trainable (1.283 %), 6 epochs, 7.6 minutes on a verified **NVIDIA RTX 2050
  (4 GB)**, peak 2,219.2 MB VRAM. Train loss 2.4049 → 1.2647; validation 2.5688 → 2.3774.
- The **12.3 MB adapter exists on disk**, reloads in a fresh process, and measurably changes
  model output (max |Δprobability| 0.115805 on the same input) — so it is not a saved no-op.
- **Baseline and adapted were both evaluated** on the dataset's own held-out `test` split
  (n=2,053, never used for training/validation/model selection) with identical preprocessing:
  mean recall **0.0115 → 0.0791** (6.9×), median rank **518 → 85** (i2t) and **645 → 84**
  (t2i). All six recall metrics improved. Chance R@1 is 0.00049.
- The adapter is **wired into the real serving workflow**: separate registry entries
  (`vqa_adapted`, `captioning_adapted`), planner selection with a recorded reason, model
  provenance in the execution trace, and the adapted version string in evidence.
  `scripts/verify_phase5.py` proves this end to end through the API — **25/25 checks pass**.

**What is deliberately *not* claimed.** Grounding was not adapted and is not reported as
improved. No VQA or captioning **accuracy** was benchmarked — retrieval improving is a
different claim, and `docs/evaluation.md` marks every unmeasured capability
`NOT EVALUATED`. Registry status stays `degraded`, not `available`. No optical–SAR fusion was
learned (Sentinel-1 was present and unused). Provenance caveats are documented rather than
smoothed over: the **imagery came from an unofficial third-party mirror** (validated against
the official annotations with 0 mismatches), the captions are **CORINE-derived and not
human-verified**, and **84 % of them are truncated** by CLIP's 77-token limit.

**Still not answerable:** temporal change (R11/R12) and optical–SAR fusion (R13) — Phases 6–7.
They route correctly and report `unavailable` rather than inventing an answer.

## Status summary (end of Phase 4)
- **Newly working**: single-image VQA (mandatory), plus **both** additional single-image
  capabilities — captioning and grounding — with real CPU inference, measured model scores,
  rendered overlay evidence, and real geographic coordinates for detected regions.
- **Honest capability status**: these run on zero-shot baselines and are registered as
  `degraded`, not `available`. They are real but uncalibrated and not yet benchmarked
  (Phase 9) or LoRA-adapted on remote-sensing data (Phase 5). See `docs/models.md`.
- **Anti-fabrication behaviours now enforced**: counting questions refused with no digits in
  the answer; captions labelled retrieval-composed rather than generated; grounding null
  results reported as inconclusive rather than as absence; coarse tile regions labelled as
  estimates, never as object outlines; missing weights reported as `unavailable`.
- **Still routed-but-unanswerable**: temporal change (R11/R12), change VQA, optical-SAR
  fusion (R13) — Phases 6–7.

## Status summary (end of Phase 3)
- **Done**: natural-language queries, optical + SAR ingestion, GeoTIFF/TIFF, input and
  metadata validation, pair-compatibility validation, upload security, previews,
  agentic orchestration, automatic task/tool selection, workflow sequencing, controlled
  parameters, confidence, execution trace.
- **Framework done, awaiting specialists**: evidence (metadata/preview/comparison items
  work; boxes/masks/overlays need Phase 4+), text+spatial output combination.
- **Routed but not answerable yet**: VQA (R7), grounding/captioning (R8), temporal change
  (R11/R12), optical-SAR fusion (R13). Each is classified correctly, planned against a
  registered tool, and returns `unavailable` naming the delivering phase. **No fabricated
  answers are produced.**
- **Planned**: RS VLM adaptation (R9/R10), benchmark evaluation (R22), reports (R25),
  accessibility audit (R28), hidden-set robustness (R23).
- No mandatory requirement is unmapped. Nothing is marked Done without tests.

### Distinguishing "routed" from "answerable"
A capability being reachable through the agent is not the same as it producing answers.
Today only `metadata_inspection` produces genuine analytical output end to end; the other
task types are wired, validated, and traceable but honestly report that their specialist is
not implemented. This distinction is deliberate and is enforced by tests
(`test_analysis_api.py::test_unimplemented_single_image_tasks_report_unavailable_not_fake_answers`).

## Demo index
1. Single-image VQA · 2. Grounding/captioning · 3. Temporal change description ·
4. Change-VQA · 5. Optical + SAR joint analysis · 6. Automatic agent routing.
Every demo must show: **answer + visual evidence + confidence + execution trace** on the
real production workflow (spec 17).

## Notes
- No mandatory requirement is unmapped.
- No status is marked `Done` until the feature is implemented, integrated (API+UI where
  applicable), and tested. Foundation items reach `Done` at Phase 1 verification.
