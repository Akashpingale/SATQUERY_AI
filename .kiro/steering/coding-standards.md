---
inclusion: always
---

# Coding Standards

## Python (backend, ml, geospatial)
- Python 3.11+. **Type hints everywhere**; validate external data with Pydantic.
- Formatting/linting: **ruff** (lint + format). Keep functions small and focused.
- Structured logging via the shared logger; **never print secrets or full stack
  traces to clients**. Return sanitized error responses.
- Explicit error handling; raise typed/domain errors, map to HTTP at the API edge.
- No hard-coded secrets, paths, or model names — read from config/registry.
- Docstrings on public functions/classes explaining contracts, not restating code.
- Keep modules cohesive; avoid files over a few hundred lines — split by responsibility.

## TypeScript (frontend)
- **Strict** TypeScript (`strict: true`). No implicit `any`.
- Function components + hooks. Reusable components; shared API layer (no `fetch`
  scattered in components — use the API client + TanStack Query).
- Validate forms/inputs with Zod. Handle loading / empty / error states explicitly.
- No secrets in the frontend bundle. Only `VITE_`-prefixed public config.

## General
- Small, reviewable changes. Match existing patterns and libraries already in the repo.
- Every major capability ships with tests (see `testing.md`).
- Prefer clarity over cleverness; comment the "why", not the "what".
