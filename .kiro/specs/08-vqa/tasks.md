# Spec 08 — Single-Image VQA · Tasks

- [x] 1. Model feasibility gate: ungated, CPU-runnable, remote-sensing-relevant model
      selected and measured (`scripts/check_models.py`)
- [x] 2. Remote-sensing concept vocabulary (24 concepts, triggers, land-cover flags)
- [x] 3. Question analysis: presence / dominant / what-visible / count / unsupported
- [x] 4. CLIP zero-shot scoring with prompt-template ensembling
- [x] 5. Text-embedding cache (warm latency ~3.0 s -> ~0.06 s)
- [x] 6. VQA specialist with a strategy per question kind
- [x] 7. Counting refusal with no digits in the answer and `Unable to determine` confidence
- [x] 8. Evidence: ranked scores, model provenance, preview
- [x] 9. Uncertainty surfacing: weak-score and mixed-scene warnings, uncalibrated caveat
- [x] 10. Registry entry with real version/checkpoint/license and honest `degraded` status
- [x] 11. `ModelUnavailableError` -> `unavailable` outcome, never a fabricated answer
- [x] 12. Tests (model-free + real inference, offline-safe)
- [x] 13. Frontend rendering of score bars and provenance

## Remaining
- [x] Swap in the LoRA-adapted model — done in Phase 5 as a separate registry entry
      (`vqa_adapted`), selected by the planner with the reason recorded in the trace
- [ ] Benchmark accuracy on a real VQA dataset (RSVQA) — Phase 9 evaluation
- [ ] Flip registry status from `degraded` once task accuracy is established — Phase 9.
      Phase 5 measured *retrieval* improvement, not VQA accuracy, so the status stays honest.
- [ ] Confidence calibration against labelled data — Phase 9
