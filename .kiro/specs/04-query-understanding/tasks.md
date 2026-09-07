# Spec 04 — Query Understanding · Tasks

- [x] 1. Task taxonomy sourced from the task registry (8 types incl. `unsupported`)
- [x] 2. `sanitize_query`: NFKC, control-char strip, length bound, injection detection
- [x] 3. `classify`: wording signals + input-shape signals + feasibility filter
- [x] 4. Routing score with runner-up margin discount
- [x] 5. `extract_target` for grounding queries
- [x] 6. `understand()` entry point returning intent + clean query + blocked signals
- [x] 7. Tests: routing accuracy, shape constraints, injection, target extraction

## Deferred
- Learned/LLM-assisted classification (interface already allows substitution). Revisit if
  hidden-set evaluation in Phase 9 shows rule coverage gaps.
- Multilingual queries.
