# Phase 12 — Gap Analysis (P0 / P1 / P2)

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-06

Prioritized against the actual SIH26167 requirements and current architecture. **No item recommends retraining a model by default.** Each answers: does SIH require it? is the architecture capable? would it need training? new dependencies? worth the risk before submission?

---

## P0 — Must fix before SIH submission

**None.** All mandatory SIH capabilities are implemented, executed, and demonstrated on the real workflow (see compliance matrix rows 1, 3, 4/5, 6–10, 11–16, 17). The regression suite is green (backend 447, frontend 126, build clean, Playwright 6/6). There is no correctness or integrity blocker for submission.

> The only hard-blocked item (VRSBench grounding IoU) is a **quantitative nice-to-have**, not a mandatory capability — grounding itself is wired and demonstrated. It is therefore P1, not P0.

---

## P1 — Strongly recommended (before or shortly after submission)

### P1-1 — Provision VRSBench images and run the grounding IoU eval
- **SIH requires it?** Grounding: yes (functionally satisfied). A *public IoU number*: strengthens evidence, not strictly required.
- **Architecture capable?** Yes — OWL-ViT + box IoU@0.5; annotations already present.
- **Training?** No. **New deps?** Only the ~3.98 GB image archive (241 GB free). **Risk?** Low; read-only eval, no threshold tuning on test.
- **Action:** follow `PHASE12_VRSBENCH_GROUNDING_BLOCKER.md`.

### P1-2 — Complete a human NVDA screen-reader pass
- **SIH requires it?** Accessibility is mandatory; automated axe (13 states, 0 violations) is done, but a real screen-reader pass is outstanding.
- **Architecture capable?** Yes — semantics/live-regions/labels already in place.
- **Training?** No. **New deps?** NVDA + a Windows audio session. **Risk?** None to code.
- **Action:** run the documented procedure in `docs/accessibility.md`; record results. Until then keep **"HUMAN ACCESSIBILITY VERIFICATION: PENDING"**.

### P1-3 — Add auth before any public/network deployment
- **SIH requires it?** Not for the local demo; the API endpoints are intentionally unauthenticated for the evaluation demo and are **explicitly flagged** as such in code and `docs/security.md`.
- **Architecture capable?** Yes. **Training?** No. **New deps?** An auth layer (e.g. API key/JWT). **Risk?** Low; additive.
- **Action:** gate `/uploads` and `/analysis` before exposing beyond localhost. Do **not** ship the demo publicly without it.

---

## P2 — Optional polish / future work

### P2-1 — Constrained/generative VQA head for public VQA benchmarks (RSVQA/VRSBench-VQA)
- **SIH requires it?** No — single-image VQA is satisfied (bounded, retrieval-based). This would only convert `INCOMPATIBLE` → measurable.
- **Architecture capable?** Yes via the registry, **but requires training** a constrained-answer or generative head, plus datasets. **Risk before submission: high** (scope + time). **Recommendation: defer.**

### P2-2 — Generative captioning
- **SIH requires it?** No — captioning is satisfied via retrieval composition (clearly labeled, not called generative).
- **Training?** Yes (a caption decoder). **Recommendation: defer**; keep the honest retrieval framing.

### P2-3 — Improve learned cross-sensor change generalization
- Evidence (LEVIR-CD/OSCD) shows the learned SECOND model is weak cross-sensor and the deterministic detector wins on OSCD. The deterministic route is already the robust default.
- **Training?** Yes (multi-sensor data). **Worth it before submission?** No — the default is already correct and safe. **Recommendation: defer**; do not replace the deterministic default.

### P2-4 — Widen learned-route input handling (GeoTIFF) if it is ever promoted
- The learned specialists read RGB via PIL and cannot open multi-band uint16 GeoTIFF; today that is fine because they are non-default and the deterministic route handles GeoTIFF.
- **Recommendation: only if** the learned route is ever promoted; otherwise no action.

### P2-5 — SAR-specific semantic analysis
- Current SAR handling is honest measured backscatter/roughness, not semantic classes.
- **Training?** Yes (SAR-labeled data). **Recommendation: defer**; out of mandatory scope.

### P2-6 — Docker `build`/`up` smoke on a clean machine
- `docker compose config` is valid (services: postgres, backend, frontend). A full image build was **not** run here (large ML image; time/bandwidth). **Recommendation:** run `docker compose build && up` once on a clean machine to confirm clean-checkout startup; low risk, good for judge reproducibility.

---

## One-line summary

**Ship-ready for the SIH local demo.** No P0 blockers. The two most valuable pre-submission actions are **P1-1 (VRSBench grounding IoU)** and **P1-2 (NVDA pass)** — both are non-training, low-risk, and only strengthen already-satisfied requirements.
