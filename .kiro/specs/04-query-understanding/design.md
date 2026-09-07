# Spec 04 — Query Understanding · Design (as built, Phase 3)

## Approach
Transparent keyword/pattern scoring **constrained by the actual input configuration**.
No model is used, deliberately: on a CPU-only machine a small text classifier would add
latency and opacity without beating explicit rules for a fixed eight-task taxonomy. The
interface (`understand() -> QueryIntent`) is model-agnostic, so a learned classifier can
replace the internals later without touching the orchestrator.

Module: `backend/app/agent/understanding.py`.

## Pipeline
```
raw query -> sanitize_query() -> classify(query, inputs) -> QueryIntent
```

### `sanitize_query`
Treats the query as untrusted data:
- NFKC unicode normalization (defeats lookalike/combining-character tricks).
- Control characters stripped; newlines/tabs collapsed to spaces.
- Length bounded to 1000 characters.
- Eight instruction-like patterns detected, **reported** as blocked signals, and removed
  from the text (e.g. "ignore previous instructions", "reveal your system prompt",
  "chain-of-thought", "execute … command", `rm -rf`, `drop table`).

Blocked fragments can never change behaviour: the available tools and their permitted
parameters come only from the registry, never from the query.

### `classify`
Two signal families are combined:

1. **Wording** — general vocabulary per task, chosen to generalize rather than match demo
   phrasings: change/difference/increase/compare; highlight/locate/outline/where is;
   describe/caption/summarize; fuse/combine/together/jointly; CRS/EPSG/resolution/bounds/bands.
2. **Input shape** — two co-located images strongly imply a pair task even when the wording
   is vague; an optical+SAR pair nudges fusion; a T1/T2 pair nudges temporal.

Then a **hard feasibility filter**: a task is selectable only if the supplied images can
satisfy its declared `min_images`/`max_images`/modalities from the task registry.

Scoring: `strength = best/3.5`, discounted by the margin over the runner-up, clamped to
0.05–1.0. This is a **routing** score, not answer confidence.

## Decisions worth noting
- **A change or fusion request never degrades into VQA.** If someone asks "what changed?"
  with one image, the result is `UNSUPPORTED` with "that needs 2 images", not a silent
  answer to a different question. Answering the wrong question is worse than refusing.
- **Infeasible candidates are filtered, not deleted**, so when nothing is feasible the
  system can still name what the user appeared to ask for.
- **Question vs statement disambiguates change tasks**: "has built-up increased?" →
  `change_vqa`; "describe the changes" → `temporal_change`.
- Grounding extracts its target noun phrase (`extract_target`) for the Phase 4 specialist.

## Contract
```python
understand(raw_query, inputs) -> (QueryIntent, sanitized_query, blocked_signals)

QueryIntent(task, score, matched_signals, target, reason, alternatives)
```

## Tests
`backend/tests/test_query_understanding.py` — routing accuracy across all task types and
both modalities, input-shape constraints (the generalization guard), target extraction,
sanitization, and four prompt-injection variants.

## Known limits
- Vocabulary is English-only.
- Rules cannot capture every phrasing; unmatched queries are reported as unsupported with
  guidance rather than guessed at. A learned classifier is the natural upgrade path if
  hidden-set evaluation shows gaps.
