# Spec 15 — Security (stub)

> Cross-cutting; hardened continuously and reviewed at finalization (Phase 10).

## Introduction
Treat all uploads and queries as untrusted. Enforce file/agent/secret safety across the
stack. See `.kiro/steering/security.md` for standing rules.

## Requirements (high level)
- Uploads: extension + MIME + size + corruption checks; safe unique filenames;
  path-traversal protection; isolated temp; resource/time limits.
- Agent: allowlisted tool/parameter schemas; reject invalid plans; never execute
  arbitrary commands; prompt-injection resistance; log blocked attempts.
- Secrets: `.env`/secret manager only; never hard-coded/committed/logged/bundled.
- Sanitized client errors; server-side detail without secrets.

## Design (outline)
`security/` validators + guards used by uploads/agent; central path-confinement helper;
input validators shared with raster ingestion (03).

## Implementation tasks (outline)
Upload validators → path confinement → parameter allowlists → injection defenses →
secret hygiene review → security tests.

## Dependencies
Backend core (01), raster ingestion (03), agent (05), API/DB (13).

## Acceptance criteria
- Malicious/invalid inputs rejected safely; no secret leakage; no arbitrary execution.

## Tests
Negative tests: malicious filename, path traversal, oversized/corrupt files,
prompt injection, invalid agent plan.

## User inputs required
- None (deployment auth decisions at Phase 10).

## Known risks
- Network-exposed endpoints without auth must be flagged; add auth if deployed publicly.
