# Spec 17 — Final Demo & Judge Readiness (stub)

> Phase 10. Stub now; full `design.md` + `tasks.md` authored when the phase begins.

## Introduction
Curated demo cases and a judge walkthrough that exercise the REAL production workflow
end-to-end, plus hidden-set generalization readiness.

## Requirements (high level)
- Six demos on the real workflow: (1) single-image VQA, (2) grounding/captioning,
  (3) temporal change description, (4) change-VQA, (5) optical+SAR joint analysis,
  (6) automatic agent routing.
- Each demo shows: answer + visual evidence + confidence + execution trace.
- No fakes: remove any mock responses, hard-coded answers/coordinates/metrics.
- Hidden-set readiness: no demo-only routing; configurable models; unseen inputs work.

## Design (outline)
`/demo` gallery driving real API calls; demo fixtures live in `data/demo/`; walkthrough
script in `docs/demo.md`.

## Implementation tasks (outline)
Curate demo inputs → wire gallery to real workflow → verify evidence/confidence/trace →
remove all mocks → hidden-set robustness checks → walkthrough doc.

## Dependencies
All feature specs (03–14) integrated; security (15); docker (16).

## Acceptance criteria
- All six demos run on the real backend; each shows answer+evidence+confidence+trace.
- No fabricated outputs anywhere in the final path.

## Tests
E2E (Playwright) for each demo; regression that no mock/hard-coded paths remain.

## User inputs required
- Official SIH/ISRO sample data if provided; manual NVDA/accessibility verification.

## Known risks
- Demo overfitting; ensure generalization. Compute limits for live inference — plan model
  sizes accordingly (cloud GPU where needed).
