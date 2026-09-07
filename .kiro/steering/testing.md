---
inclusion: always
---

# Testing Rules

Every major feature ships with tests. "Code exists" is never "done".

## Backend (pytest)
- **Unit**: raster/metadata/CRS/alignment validation, query classification, routing,
  parameter validation, evidence, confidence, API schemas.
- **Integration**: upload -> validation -> analysis; registry; database; evidence; reports.
- **Contract**: each specialist's input/output contract.
- Tests run **without a GPU** and **without a running Postgres** (use SQLite) so CI and
  local dev are fast and reliable.

## Frontend
- **Vitest + React Testing Library** for components, hooks, states (loading/empty/error).
- **Playwright** for critical browser flows: upload, query, analysis, result, evidence,
  trace, history, report, and error handling.

## Negative tests (required)
Invalid/corrupted files, unsupported formats, incompatible pairs, mismatched CRS,
missing modality, missing model, GPU-unavailable, timeout, invalid agent plan,
malicious filename, and prompt injection.

## Discipline
- Add/extend tests in the same change as the feature.
- Run the relevant suite before declaring a task complete; report real results.
- Never assert against fabricated model outputs; use fixtures or clearly-labeled mocks.
