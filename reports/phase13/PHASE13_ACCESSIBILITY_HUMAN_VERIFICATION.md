# Phase 13 — Human Accessibility (NVDA) Verification

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07

## Headline status

> **HUMAN NVDA VERIFICATION = NOT COMPLETED**

A real NVDA screen-reader pass requires a human tester on a Windows machine with an audio session. This phase was executed by an automated agent with **no human tester and no screen-reader/audio environment**, so a genuine NVDA pass could not be performed. It is **not** simulated, and **no PASS is claimed**. This matches the Phase 12 status (PENDING) and remains a P1 release item.

**Automated and code-level accessibility are reported separately below and must not be conflated with human screen-reader verification.**

## Automated accessibility (separate, actually run)

- **axe-core (via vitest-axe):** **13 states, 0 violations** — `frontend/tests/accessibility.test.tsx`. Re-run this phase as part of the 126/126 vitest suite.
- One rule excluded with reason: `color-contrast` (jsdom cannot compute layout/CSS variables); contrast verified manually against the palette. No other rule disabled.
- Command: `cd frontend; npx vitest run tests/accessibility.test.tsx`.

## Code-level accessibility (inspection, not screen-reader)

From `docs/accessibility.md`, verified present in components: `focus-visible:ring-2` on interactive controls; labelled query textarea with `aria-describedby`; native `<button>` example chips; `role="tablist"`/`role="tab"`/`aria-selected` visualization switch; native `<details>/<summary>` disclosures for evidence/trace; `aria-live="polite"` + `aria-busy` result region; `role="alert"` errors; meaningful `alt` on images; decorative icons `aria-hidden`. This is **code-level evidence**, not a screen-reader announcement test.

## NVDA test plan (ready for a human tester)

Environment: NVDA on Windows; frontend `http://localhost:5173`, backend `http://localhost:8000`. Keyboard-only. Classify each: **PASS / FAIL / PARTIAL / NOT_TESTED**.

| # | Item | Screen-reader status | Notes |
|---|---|---|---|
| 1 | Page title | NOT_TESTED | requires human NVDA |
| 2 | Main navigation / landmarks | NOT_TESTED | code has landmarks |
| 3 | Input workspace region | NOT_TESTED | |
| 4 | Optical upload control | NOT_TESTED | labelled in code |
| 5 | SAR upload control | NOT_TESTED | |
| 6 | Temporal T1 upload | NOT_TESTED | |
| 7 | Temporal T2 upload | NOT_TESTED | |
| 8 | Single-image upload | NOT_TESTED | |
| 9 | Query input | NOT_TESTED | labelled + described in code |
| 10 | Analyze button | NOT_TESTED | native submit button |
| 11 | Example questions | NOT_TESTED | native buttons |
| 12 | Analysis pipeline / progress | NOT_TESTED | aria-live in code |
| 13 | Evidence viewer | NOT_TESTED | details/summary |
| 14 | History | NOT_TESTED | |
| 15 | Status indicators | NOT_TESTED | text+icon, not colour-only |
| 16 | Error messages | NOT_TESTED | role="alert" in code |
| 17 | Validation messages | NOT_TESTED | |
| 18 | Results / confidence | NOT_TESTED | aria-live region |
| 19 | Modal/dialog behavior | NOT_TESTED | |
| 20 | Keyboard-only navigation | NOT_TESTED | focus-visible in code |

### Screen-reader checks (per component) — all NOT_TESTED

accessible name · role · state · keyboard access · focus order · focus visibility · meaningful announcement · error announcement · disabled-state announcement · dynamic result announcement — for upload controls, drag/drop areas, buttons, tabs, expandable sections, dialogs, evidence viewer, analysis progress, confidence/result status. **None verified with a screen reader.**

### Keyboard-only pass — NOT_TESTED (human)

Tab / Shift+Tab / Enter / Space / Escape / Arrows: no keyboard trap, logical focus order, all controls reachable, visible focus, dialogs exitable, upload + evidence viewer usable. Code-level inspection is favorable; **not** confirmed by a human run.

## Result classification summary

| Category | Result |
|---|---|
| Automated axe (13 states) | **PASS (0 violations)** |
| Code-level keyboard/semantics | favorable (inspection only) |
| Human NVDA screen-reader | **NOT_TESTED / NOT COMPLETED** |
| Human keyboard-only pass | **NOT_TESTED / NOT COMPLETED** |

**Accessibility overall = PARTIAL** (automated done; human verification outstanding). To complete, a human runs the plan above and records PASS/FAIL/PARTIAL per row in `docs/accessibility.md` and here.
