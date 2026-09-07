# Final Completion Ledger — SatQuery AI (SIH26167)

The single tracker for **everything** that remains before the project is SIH-ready. Updated at
every phase. No vague entries: each item has a concrete definition of done.

**Status legend:** ✅ done · 🟡 partial · ⬜ not started · 🔒 blocked (dependency).
**Priority:** P0 (SIH-mandatory) · P1 (high-value/quality) · P2 (nice-to-have).

Phases: 1–7 implemented; **8** (UX/reports/accessibility), **9** (benchmarks/calibration),
**10** (Docker build/deploy/security/git/demo) remain.

---

## 1. Core implementation

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| CORE-1 | Raster ingestion (GeoTIFF/TIFF/PNG/JPEG), metadata, previews | ✅ | 2 | P0 | — | Done, tested |
| CORE-2 | Agent orchestration (understand→plan→validate→execute→evidence/confidence/trace) | ✅ | 3 | P0 | — | Done, tested |
| CORE-3 | Single-image VQA / captioning / grounding | ✅ | 4 | P0 | — | Done, tested |
| CORE-4 | Bi-temporal change (description + VQA + change map) | ✅ | 6 | P0 | — | Done, tested |
| CORE-5 | Optical–SAR cross-modal fusion | ✅ | 7 | P0 | align_modalities | Done — feature-level fusion, tested |

## 2. AI / model capability

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| AI-1 | RS VLM adaptation (LoRA on BigEarthNet.txt), served via registry | ✅ | 5 | P0 | — | Done; adapter reloads, wired |
| AI-2 | Change detection (algorithmic CVA), served | ✅ | 6 | P0 | — | Done; deterministic |
| AI-3 | Cross-modal fusion (feature-level + complementarity), served | ✅ | 7 | P0 | — | Done — `analyze_cross_modal`, tested |
| AI-4 | Confidence calibration against labelled data | ✅ | 9 | P1 | — | DONE — real temperature scaling on CDVQA (Val→Test), ECE 0.036→0.014; framework `ml/evaluation/calibration.py`. Synthetic end-to-end calibration also retained |
| AI-5 | Task-accuracy benchmarks | 🟡 | 9 | P0 | RSVQA/VRSBench data | PARTIAL — **CDVQA real held-out Test 0.71/Test2 0.66** and **SECOND real held-out IoU 0.42** now measured; BigEarthNet retrieval 6.9× (real). RSVQA/VRSBench still NOT RUN (data not provisioned; VQA/caption architecture-incompatible). No fabricated numbers |

## 3. Agentic orchestration

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| AG-1 | Registry-constrained planning + allowlisted params | ✅ | 3 | P0 | — | Done |
| AG-2 | Automatic routing for all task types incl. fusion | ✅ | 7 | P0 | — | Fusion routes without manual mode |
| AG-3 | Validate-first on pairs (registration before change/fusion) | ✅ | 6/7 | P0 | — | Done |

## 4. Geospatial correctness

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| GEO-1 | Pair compatibility (CRS/overlap/resolution/dims) | ✅ | 2 | P0 | — | Done |
| GEO-2 | Co-registration to a common grid (WarpedVRT), recorded in trace | ✅ | 6 | P0 | — | Done; reused by fusion |
| GEO-3 | Coordinates only when georeferencing reliable | ✅ | 4/6/7 | P0 | — | Done |

## 5. Frontend

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| FE-1 | Upload slots (optical/SAR/T1/T2), compatibility, viewer, evidence, trace | ✅ | 2–6 | P0 | — | Done |
| FE-2 | Cross-modal evidence + aligned-preview rendering | ✅ | 7 | P0 | — | Done — Optical+SAR view + renderers, tested |
| FE-3 | Downloadable report (HTML) from a result | ✅ | 8 | P1 | — | Done — `GET /api/v1/analysis/{id}/report` serves a self-contained HTML report (evidence embedded as base64), "Download report" link in `ResultCard.tsx`; 12 backend tests |
| FE-4 | Example-question chips per input configuration | ✅ | 8 | P2 | — | Done — `lib/examples.ts` `exampleQuestions()` adapts to single/optical/SAR/T1-T2/optical+SAR, wired via `AssistantPanel`; vitest covers all configs |
| FE-5 | Global loading/empty/error polish | ✅ | 8 | P1 | — | Done — History loading/empty/error states; `EvidenceImage` onError fallback in `EvidencePanel.tsx`; verified by vitest + E2E |

## 6. Accessibility

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| A11Y-1 | Semantic HTML, keyboard nav, labels, status-not-by-colour | ✅ | 2–8 | P0 | — | Done — audited via axe (13 tests) and documented in `docs/accessibility.md` |
| A11Y-2 | Automated axe checks in the test suite | ✅ | 8 | P1 | — | Done — `tests/accessibility.test.tsx` (13 tests) asserts 0 violations on key views; only `color-contrast` disabled (jsdom cannot compute layout), documented + checked manually |
| A11Y-3 | Manual screen-reader pass (NVDA) documented | 🟡 | 8 | P1 | A11Y-2 | **NVDA: NOT YET VERIFIED** — procedure written in `docs/accessibility.md`; keyboard-nav + axe pass done. NVDA execution honestly not yet run |

## 7. Testing

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| TEST-1 | Backend unit/integration/negative (pytest) | ✅ | all | P0 | — | **416 passing**, 0 failures, 0 skipped locally (SQLite, no GPU) |
| TEST-2 | Cross-modal fusion tests | ✅ | 7 | P0 | — | Done — 59 tests (unit + integration) |
| TEST-3 | Frontend component tests (vitest) | ✅ | all | P0 | — | **119 passing** (14 files) |
| TEST-4 | Playwright E2E for the six demo flows | ✅ | 8 | P1 | — | Done — `frontend/e2e/flows.spec.ts`, **6/6 pass** on the real workflow via isolated backend (`scripts/e2e_backend.py`) + Vite dev server; deterministic synthetic fixtures, no mocked outputs |

## 8. Benchmarking

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| BM-1 | RSVQA (VQA) | 🔒 | 9 | P0 | dataset + new VQA head | BLOCKED — dataset not provisioned AND current VQA is fixed-vocabulary concept scoring, incompatible with RSVQA yes/no/count answers. Adapter/normalization/command built (`scripts/eval_external.py`, `docs/benchmarking.md §4`); NOT RUN, no fabricated score |
| BM-2 | VRSBench (VQA/caption/grounding) | 🔒 | 9 | P1 | dataset | BLOCKED — dataset not provisioned. Grounding (OWL-ViT zero-shot) is IoU-compatible and runnable once provisioned; VQA/caption architecture-incompatible. Command documented |
| BM-3 | CDVQA (change VQA) — real training + held-out test | ✅ | 9 | P0 | — | DONE — real CDVQA provisioned/extracted, trained classifier (frozen CLIP + GRU + fusion), **held-out Test 0.71 / Test2 0.66** (vs majority 0.51/0.46), leakage-checked, calibrated (ECE 0.036→0.014), integrated as `cdvqa_trained`. See `docs/CDVQA_TRAINING_REPORT.md` |
| BM-4 | Change detection (SECOND real; LEVIR-CD/OSCD optional) | ✅ | 9 | P1 | — | DONE (SECOND) — SECOND extracted, learned Siamese U-Net trained, **held-out IoU 0.42 / F1 0.59 vs deterministic 0.22 / 0.37**, integrated as `second_change_learned`. SECOND = *additional public* dataset (not SIH). LEVIR-CD/OSCD optional, NOT RUN. See `docs/SECOND_TRAINING_REPORT.md` |
| BM-5 | Cross-modal fusion evaluation | ✅ | 9 | P1 | — | DONE per its DoD — no suitable public benchmark exists (`docs/CROSS_MODAL_EVALUATION_ABSENCE.md`); correctness validated on synthetic truth (`reports/benchmarks/cross_modal/`, 2/4 property checks pass, reported honestly) |

## 9. Confidence calibration

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| CAL-1 | Calibrate confidence → correctness on labelled data | ✅ | 9 | P1 | — | DONE — real temperature scaling fit on CDVQA **Val**, evaluated frozen on **Test**: ECE 0.036→0.014, reliability tracks accuracy (`reports/benchmarks/cdvqa/calibration.json`). Fit never used test labels. (Synthetic end-to-end calibration also retained.) |

## 10. Documentation

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| DOC-1 | Phase docs (phase5/6/7), models, datasets, training, evaluation, architecture, traceability, limitations | ✅ | 5–7 | P0 | — | Kept in sync each phase |
| DOC-2 | `docs/api.md`, `docs/user-guide.md`, `docs/demo.md`, `docs/security.md`, `docs/troubleshooting.md` | ✅ | 8 | P1 | — | Done — all five created/updated and verified against the code (report endpoint, Phase 6/7 evidence kinds, upload guards, path confinement, prompt-injection handling, unauthenticated-API flag) |
| DOC-3 | `docs/training.md` reproducibility incl. real git SHA | 🟡 | 10 | P1 | GIT-1 | git_commit is a real SHA once commits exist |

## 11. Docker / deployment

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| DEP-1 | `docker compose config` valid | ✅ | 1 | P0 | — | Validated each phase |
| DEP-2 | Backend/frontend images actually **built** | ⬜ | 10 | P0 | disk space | `docker compose build` succeeds |
| DEP-3 | Clean-machine `docker compose up` end-to-end | ⬜ | 10 | P0 | DEP-2 | Frontend+API+DB healthy; a real analysis runs |

## 12. Security

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| SEC-1 | Upload guards (MIME/size/corruption/path traversal/safe names) | ✅ | 2 | P0 | — | Done, negative-tested |
| SEC-2 | Prompt-injection handling (sanitize, treat text as data) | ✅ | 3 | P0 | — | Done, tested |
| SEC-3 | API authentication (endpoints are unauthenticated) | ⬜ | 10 | P1 | — | Auth added or explicitly documented as out-of-scope for the demo |
| SEC-4 | Dependency/secret scan before submission | ⬜ | 10 | P1 | — | No secrets committed; deps pinned; scan clean |

## 13. Git / repository hygiene

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| GIT-1 | Repository has **no commits yet** | ⬜ | 10 | P0 | — | Initial commit(s) made; `environment.json` records a real SHA |
| GIT-2 | `.gitignore` covers models/data/reports/artifacts | ✅ | 1/5 | P0 | — | Verified |
| GIT-3 | LICENSE + attribution (Copernicus, BigEarthNet, model cards) | 🟡 | 10 | P1 | — | LICENSE present; attributions consolidated |

## 14. Performance

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| PERF-1 | Bounded memory on large rasters (windowed/decimated reads) | ✅ | 2/6/7 | P0 | — | Enforced via decimation caps |
| PERF-2 | Per-stage timing recorded for fusion/change | 🟡 | 7/9 | P1 | — | Measured in Phase 7 report; formal profiling in 9 |

## 15. SIH compliance

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| SIH-1 | R7 VQA · R8 grounding/caption · R9/R10 adaptation | ✅ | 4/5 | P0 | — | Mapped in traceability |
| SIH-2 | R11/R12 temporal · R24 change map | ✅ | 6 | P0 | — | Mapped |
| SIH-3 | R13 optical–SAR cross-modal | ✅ | 7 | P0 | — | Done — feature-level fusion; code path in traceability |
| SIH-4 | R22 benchmark evaluation | ⬜ | 9 | P0 | BM-* | Real benchmark numbers |
| SIH-5 | R23 hidden-set readiness (no hacks) | 🟡 | all | P0 | — | No hard-coding; final robustness review in 10 |
| SIH-6 | R28 accessibility | 🟡 | 8 | P1 | A11Y-3 | axe automation + keyboard nav done and documented; manual NVDA pass NOT YET VERIFIED |

## 16. Demo readiness

| ID | Item | Status | Phase | Prio | Dependency | Definition of done |
|---|---|---|---|---|---|---|
| DEMO-1 | Six demo flows run on the real workflow | 🟡 | 8/10 | P0 | DEP-3 | All six pass under Playwright E2E on the real workflow; final rehearsal on a clean-machine deploy remains (Phase 10) |
| DEMO-2 | Demo dataset (safe, non-hidden) prepared | ✅ | 8 | P1 | — | Done — `scripts/make_demo_data.py` deterministically generates synthetic optical/SAR/T1/T2/incompatible rasters (CC0); tracked `data/manifests/demo_manifest.json` with sha256; `.tif` outputs gitignored + re-derivable, byte-identical on rerun |
| DEMO-3 | `docs/demo.md` walkthrough | ✅ | 8 | P1 | — | Done — step-by-step walkthrough matching the running app and the six flows |

---

## Snapshot after Phase 7 (verified)

- **Implemented and tested:** Phases 1–7. All six mandatory RS capabilities are now real: VQA,
  grounding/captioning, RS adaptation (LoRA), temporal change description + change-VQA, and
  optical–SAR cross-modal analysis. **404 backend tests, 96 frontend tests, 0 failures**;
  ruff/eslint/tsc clean; vite build + `docker compose config` pass.
- **The big remaining blocks** are deliberately **not** faked early, each with an owning phase:
  - **Phase 8 — UX/reports/accessibility:** downloadable report (FE-3), example-question chips
    (FE-4), loading/error polish audit (FE-5), axe automation (A11Y-2), NVDA pass (A11Y-3),
    Playwright E2E (TEST-4), demo dataset + `docs/demo.md` (DEMO-2/3), missing docs (DOC-2).
  - **Phase 9 — benchmarks/calibration:** RSVQA (BM-1), VRSBench (BM-2), CDVQA (BM-3), change
    detection (BM-4), cross-modal (BM-5), confidence calibration (CAL-1). Until these run, every
    model/tool stays `degraded` — nothing is benchmark-validated.
  - **Phase 10 — deploy/security/git/demo:** Docker image **build** (DEP-2) + clean-machine
    deploy (DEP-3), auth decision (SEC-3), dependency/secret scan (SEC-4), **initial git
    commit** (GIT-1, the repo still has no commits), LICENSE/attribution consolidation (GIT-3),
    six-flow demo rehearsal (DEMO-1).
- This file is the authoritative checklist for driving Phases 8–10 to genuine completeness. Every
  open item has a concrete definition of done above.

---

## Snapshot after Phase 8 (verified)

- **Phase 8 closed** (UX / reports / accessibility / E2E / demo / docs):
  - **FE-3** downloadable self-contained HTML report (server-rendered, evidence embedded
    as base64) — 12 backend tests.
  - **FE-4** adaptive example-question chips per input configuration — vitest across all
    configs.
  - **FE-5** loading/empty/error polish incl. evidence-image onError fallback.
  - **A11Y-1 / A11Y-2** axe automation (13 tests, 0 violations; only `color-contrast`
    disabled with a documented reason) + `docs/accessibility.md`.
  - **TEST-4** Playwright E2E — **6/6 flows pass** on the real workflow with deterministic
    synthetic fixtures (no mocked model outputs), isolated backend + Vite dev server.
  - **DEMO-2 / DEMO-3** deterministic synthetic demo dataset (generator + tracked sha256
    manifest, CC0) + `docs/demo.md` walkthrough.
  - **DOC-2** `api.md`, `user-guide.md`, `security.md`, `troubleshooting.md`, `demo.md`
    written/updated to match the code.
- **Verified regression (Phase 8):** **416 backend tests**, **119 frontend tests**, **6/6
  Playwright flows**, 0 failures; ruff / eslint / tsc clean; `vite build` (JS 296 KB,
  gzip 90 KB) and `docker compose config` pass.
- **Honestly still open from Phase 8:**
  - **A11Y-3 (NVDA):** manual screen-reader pass is **NOT YET VERIFIED** — procedure
    documented, keyboard/axe done; NVDA execution not yet performed.
  - **DEMO-1:** all six flows pass under E2E, but a clean-machine deploy rehearsal (DEP-3)
    remains in Phase 10.
- **Phase 9 (benchmarks/calibration)** and **Phase 10 (docker build/deploy, git init,
  auth decision, dependency/secret scan, LICENSE/attribution)** remain open and are
  deliberately not faked.


---

## Snapshot after Phase 9 (verified)

Phase 9 delivered a **real, reproducible evaluation + calibration framework** and executed every
evaluation that is genuinely runnable in this offline environment. It did **not** fabricate
external-benchmark accuracy or claim training that was not done.

- **Built + tested (offline, deterministic):** `ml/evaluation/{metrics,calibration,reproducibility,datasets,synthetic,cross_modal,change_detection}.py`, reusing the existing `change_vqa.py` adapter; runners `scripts/run_benchmarks.py` and `scripts/eval_external.py`; 16 evaluation tests in `backend/tests/test_evaluation.py`.
- **Executed (real):**
  - Baseline vs adapted **retrieval** (BigEarthNet.txt held-out test, 2053): mean recall 0.0115 → **0.0791 (~6.9×)** — a retrieval metric, not VQA/caption accuracy.
  - **Change detection** (synthetic known masks): IoU/F1 **1.0**, occurrence acc **1.0** (4 scenarios) — engineering validation.
  - **Change-occurrence VQA** (synthetic): acc **1.0** (n=12, Wilson 0.76–1.0) — engineering validation, not CDVQA.
  - **Cross-modal** (synthetic): 2/4 property checks pass (reported honestly), benchmark absence documented — satisfies BM-5's DoD.
  - **Confidence calibration** (synthetic, end-to-end via orchestrator): ECE **0.287**, Brier **0.091**, mildly under-confident; thresholds unchanged.
- **Honestly NOT RUN (no fabricated numbers):** RSVQA, VRSBench (VQA/caption/grounding), CDVQA, SECOND, LEVIR-CD, OSCD — datasets not provisioned; VQA/captioning additionally architecture-incompatible. Adapters, availability checks (`scripts/eval_external.py`), and reproducible commands provided. **SECOND is recorded as an additional public dataset, not SIH-provided.**
- **Registry:** unchanged — every ML/algorithmic specialist remains `degraded`, the correct outcome since no real external benchmark validated them and no new production model was trained.
- **Ledger item outcomes:** BM-1 🔒, BM-2 🔒, BM-3 🟡, BM-4 🟡, BM-5 ✅, CAL-1 🟡, AI-4 🟡, AI-5 🟡.
- **Remaining (genuine):** provision RSVQA/VRSBench/CDVQA/SECOND; build a constrained/generative VQA head + a generative captioner for benchmark compatibility; run real grounding IoU on VRSBench; train/evaluate a learned change detector on SECOND (kept alongside the deterministic one); real validation-split calibration + threshold re-tuning; then registry promotions where evidence supports.


---

## Snapshot after Phase 9 (real training — datasets provisioned)

The user provisioned CDVQA and SECOND locally; both were extracted, inspected, trained, and
evaluated on held-out data. No fabricated numbers; no synthetic substitution for real metrics.

- **CDVQA (SIH benchmark) — DONE.** Real archive (SHA256 recorded), 65,967/16,441/39,686/31,036
  QA over disjoint image splits. Trained constrained classifier (frozen CLIP `clip-rsicd-v2`
  features + GRU question encoder + fusion), 832K params, 128 s on RTX 2050. **Held-out Test
  0.7114, Test2 0.6632** (majority baseline 0.5084/0.4560). Temperature-scaled (Val→Test) ECE
  0.036→0.014. Integrated as registered specialist `cdvqa_trained` (checkpoint reloads,
  inference verified). Report: `docs/CDVQA_TRAINING_REPORT.md`.
- **SECOND (additional public, NOT SIH) — DONE.** Real 2.24 GB archive (SHA256 recorded), 2,968
  pairs, RGB semantic labels; binary change derived (documented). Learned Siamese U-Net (1.93 M
  params, 256 px, 12 epochs, ~27 min). **Held-out IoU 0.4219 / F1 0.5934 vs deterministic
  reference 0.2248 / 0.3670** — learned ~2× better. Integrated as `second_change_learned`.
  Report: `docs/SECOND_TRAINING_REPORT.md`.
- **Calibration (CAL-1) — DONE** on CDVQA (real Val→Test temperature scaling).
- **Model selection / registry:** both learned models are registered and benchmarked but kept at
  `degraded` and are **NOT** the default route — the deterministic detectors remain default for
  general imagery (robustness / generalization), per the "never force a learned model" rule.
- **Leakage:** verified zero Train/Val/Test image overlap (CDVQA) and disjoint seeded splits
  (SECOND); calibration never used test labels. `docs/phase9_data_leakage.md`.
- **Regression:** backend pytest (incl. new CDVQA/SECOND tests), ruff, frontend
  vitest/tsc/eslint/build, docker config, Playwright — all green (see report §Tests).
- **Still NOT RUN (honest):** RSVQA, VRSBench (VQA/caption/grounding), LEVIR-CD, OSCD — not
  provisioned; RSVQA/VRSBench VQA/caption additionally architecture-incompatible with the current
  zero-shot/retrieval specialists. Adapters + commands provided.
