# Spec 04 — Query Understanding (stub)

> Phase 3. Stub now; full `design.md` + `tasks.md` authored when the phase begins.

## Introduction
Interpret the user's natural-language query and the current input configuration into a
structured, validated task intent the agent can act on.

## Requirements (high level)
- Classify query into a supported task type (VQA, captioning, grounding, temporal
  change, change-VQA, optical-SAR fusion, metadata inspection, unsupported).
- Combine query with detected modalities/image-count to disambiguate intent.
- Produce a structured intent with confidence; treat text as untrusted (no injection).

## Design (outline)
`agent/understanding.py`: rule + lightweight-model hybrid classifier producing a typed
`QueryIntent`. Inputs from raster metadata (spec 03) and the task registry (spec 06).

## Implementation tasks (outline)
Task taxonomy → classifier → intent schema → confidence → injection-safe parsing → tests.

## Dependencies
Task registry (06), raster metadata (03). Optional small text model (CPU-friendly).

## Acceptance criteria
- Representative queries map to correct task types; ambiguous/unsupported handled gracefully.
- Modality/image-count constrains the classification (e.g. two images → temporal/fusion).

## Tests
Unit: labeled query→task fixtures; negative: unsupported/ambiguous/injection strings.

## User inputs required
- None expected (may ask about optional external LLM for parsing later).

## Known risks
- Overfitting to demo phrasings — keep taxonomy general for hidden-set readiness.
