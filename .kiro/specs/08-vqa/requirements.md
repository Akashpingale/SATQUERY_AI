# Spec 08 — Single-Image VQA (stub)

> Phase 4. Stub now; full `design.md` + `tasks.md` authored when the phase begins.

## Introduction
Mandatory single-image Visual Question Answering over remote-sensing imagery, returned
with confidence, evidence where available, model/tool info, and an execution trace.

## Requirements (high level)
- Answer natural-language questions about a single optical or SAR image
  (e.g. dominant land cover, presence of water/buildings, visible major objects).
- Uniform specialist input/output contract; registered in the model registry.
- CPU-runnable path (small/quantized) with option to use the adapted VLM (spec 07).

## Design (outline)
`ml/inference/vqa.py` implementing the specialist contract; backend wires it via the
registry and agent. Evidence/confidence via spec 12.

## Implementation tasks (outline)
Define VQA contract → baseline CPU model → registry entry → agent integration →
evidence/confidence → tests → later swap in adapted checkpoint (07).

## Dependencies
Registry (06), agent (05), raster ingestion (03), evidence/confidence (12).

## Acceptance criteria
- Given an image + question, returns a coherent answer + confidence + trace via the API.
- Swappable model without changing API/UI contracts.

## Tests
Contract test (I/O schema); integration (image+question→answer); negative (no image,
unsupported question).

## User inputs required
- HF token if the chosen baseline is gated (later).

## Known risks
- CPU baseline quality is limited pre-adaptation; document as baseline until spec 07 lands.
