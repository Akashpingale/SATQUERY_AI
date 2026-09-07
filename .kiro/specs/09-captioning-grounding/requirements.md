# Spec 09 — Captioning & Grounding (stub)

> Phase 4. Stub now; full `design.md` + `tasks.md` authored when the phase begins.

## Introduction
At least one additional single-image capability (preferably both): remote-sensing-aware
**captioning** (scene description) and **grounding** (localize a queried object/region
with box/mask + overlay).

## Requirements (high level)
- Captioning: generate a remote-sensing-aware description of a single image.
- Grounding: return bounding box and/or mask for a queried target, with a UI overlay
  and confidence.
- Both follow the uniform specialist contract and are registered.

## Design (outline)
`ml/inference/captioning.py`, `ml/inference/grounding.py`; evidence overlays via spec 12;
served through registry (06) + agent (05).

## Implementation tasks (outline)
Define contracts → CPU-runnable baselines → registry entries → agent integration →
evidence overlays → tests.

## Dependencies
Registry (06), agent (05), raster ingestion (03), evidence/confidence (12).

## Acceptance criteria
- Captioning returns a plausible RS description; grounding returns box/mask + overlay
  and confidence, via API/UI.

## Tests
Contract tests; grounding IoU/localization on small fixtures; caption sanity checks; negatives.

## User inputs required
- HF token if a chosen baseline is gated (later).

## Known risks
- Grounding quality on CPU baselines; refine with adaptation (07) where possible.
