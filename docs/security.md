# Security

This document describes the security controls implemented in SatQuery AI, and the
things a reader must know before exposing the service beyond a local development or
evaluation environment. It reflects the code as it exists today; where a control is
absent or deliberately deferred, that is stated explicitly rather than implied.

SatQuery AI treats **every uploaded image and every user query as untrusted input**.

---

## 1. Threat model (scope)

- **In scope:** malformed / malicious uploads, path traversal via filenames and stored
  paths, oversized uploads, disguised file types, prompt injection through the query or
  through text embedded in imagery, unsafe tool/parameter selection by the agent,
  accidental leakage of internal error details or secrets.
- **Out of scope (today):** authentication, authorization, multi-tenant isolation,
  rate limiting, and network hardening. The API is **unauthenticated** — see
  [Section 8](#8-network-exposure-and-authentication).

---

## 2. Upload and file validation

Implemented in `backend/app/security/files.py`, `backend/app/services/storage.py`,
and `backend/app/services/ingestion.py`, wired at the API edge in
`backend/app/api/v1/routes/uploads.py`.

- **Extension allowlist.** Supported raster extensions come from
  `geospatial.SUPPORTED_EXTENSIONS`. `sanitize_filename` drops any extension that is
  not on the allowlist rather than preserving an unknown/dangerous one.
- **MIME / content-type check (advisory).** `is_allowed_content_type` checks the
  declared `Content-Type` against `ALLOWED_CONTENT_TYPES`. A disallowed type is
  rejected with **HTTP 415**. This is intentionally advisory: the declared type can be
  spoofed, so the authoritative check is opening the bytes as a raster.
- **Authoritative content validation.** `validate_raster_file` (in `ingestion.py`)
  opens the stored bytes with the raster reader and extracts metadata. If the file is
  not a valid, readable raster, the upload is **rejected and the stored bytes are
  deleted** (`storage.delete_stored`). A file that merely has a `.tif` name but is not a
  real raster does not survive.
- **Size limit.** `save_upload_stream` streams to disk in chunks and aborts with
  `UploadTooLargeError` once the byte count exceeds `settings.max_upload_mb`
  (default **200 MB**, from `backend/app/core/config.py`, override via env). The limit is
  enforced *during* streaming, so an oversized file cannot be fully buffered first.

## 3. Safe filenames and path-traversal protection

- **Never trust the client filename.** `build_storage_name` derives the on-disk name
  from a server-generated unique id (`uuid4` hex) plus a sanitized, length-bounded stem.
  The original name is kept only as a readable hint and is never used as a path.
- **`sanitize_filename` hardening:** NFKD unicode normalization, NUL stripping,
  directory-component stripping for both `/` and `\`, collapsing of unsafe characters to
  `_`, and a 64-char stem bound. This defeats `../`, absolute paths, and unicode
  lookalike separators.
- **Path confinement.** `resolve_within(root, *parts)` resolves the candidate path and
  raises `UnsafePathError` unless it stays inside the configured storage root. **All**
  reads and writes go through `storage.absolute_path` / `resolve_within` — uploads,
  previews, evidence overlays, and report image embedding. A crafted `storage_path` or
  evidence token that would escape the root yields **404**, never a file outside the root.

## 4. Temporary files, resource, and time limits

- Uploaded rasters and derived artifacts (previews, evidence overlays) live on the
  filesystem under the storage root; PostgreSQL holds metadata only (see
  `docs/architecture.md`).
- A failed ingestion cleans up its stored bytes immediately.
- `settings.request_timeout_seconds` (default **120s**) bounds long-running analysis.
- **Known limitation:** there is no background reaper for orphaned artifacts; long-lived
  deployments should add periodic cleanup. Documented in `docs/limitations.md`.

## 5. Agent and model safety

Implemented in `backend/app/agent/` (`understanding.py`, `validation.py`,
`orchestrator.py`) and `backend/app/registry/`.

- **Registry-only tool selection.** The agent can only choose tools that exist in the
  model/tool registry. It cannot invent tools or checkpoints.
- **Allowlisted, validated parameters.** `validate_parameters` / `validate_plan` accept
  only parameters declared in each tool's `allowed_parameters` (`ParamSpec`). Unknown
  parameters are rejected; missing ones fall back to declared defaults. Invalid plans
  raise `PlanRejected`.
- **No arbitrary command execution.** User input is never turned into a shell command or
  executed. There is no `eval`/`exec`/shell path from query text to the OS.

## 6. Prompt-injection resistance

Implemented in `backend/app/agent/understanding.py` (`sanitize_query`) and surfaced in
the trace (`backend/app/agent/trace.py`).

- The query is **untrusted data**, never instructions. `sanitize_query` applies NFKC
  normalization, strips control characters, bounds length, and matches a set of
  instruction-like patterns (`_INJECTION_PATTERNS`) such as "ignore previous
  instructions", "disregard prior rules", "you are now a…", "system prompt", "reveal
  your prompt/instructions/reasoning", "chain of thought", and "execute command/shell".
- Matching fragments are **stripped before classification** and recorded as *blocked
  signals*. They can never change which tools exist or what parameters are allowed —
  that comes only from the registry.
- Blocked attempts are **logged** (`orchestrator.py`) and **surfaced in the trace** as a
  warning ("instruction-like fragment(s) were ignored; the query was treated strictly as
  an imagery question").
- Text extracted from imagery is likewise treated as data, not as instructions.
- **Chain-of-thought is never exposed.** The trace is an auditable list of steps and
  signals, not model reasoning.

## 7. Secrets

- Secrets come from `.env` / the process environment via `backend/app/core/config.py`
  (Pydantic settings) only. **No secrets are hard-coded** in source.
- `.env` is gitignored; `.env.example` documents the keys.
- Secrets are never logged and never shipped to the frontend. The frontend may only read
  `VITE_`-prefixed public configuration (see `docs/setup.md`).
- Secrets are kept out of the model/agent context.

## 8. Network exposure and authentication

> **The `/api/v1` endpoints are UNAUTHENTICATED.** This is called out directly in
> `backend/app/api/v1/routes/analysis.py`.

This is acceptable for **local development and the evaluation demo**, where the service
binds to localhost. It is **not acceptable for public deployment**. Before exposing the
service beyond localhost you must add, at minimum:

- authentication and authorization on all write and read endpoints,
- rate limiting / request quotas,
- a locked-down CORS policy (see below),
- TLS termination in front of the app.

This item is tracked as open work (SEC-3/SEC-4) and is **not** claimed as complete.

### CORS

CORS is configured in `backend/app/main.py`. Confirm the allowed origins match your
deployment; a permissive origin list combined with the lack of auth would let any site
call the API from a user's browser. For the local demo the frontend origin is `:5173`.

## 9. Errors and information exposure

- Client-facing errors are **sanitized** (typed domain errors mapped to HTTP status
  codes at the API edge). Full stack traces and internal details are **not** returned to
  clients.
- Details are logged server-side via the shared structured logger, with a per-request
  `X-Request-ID` (`main.py`) for correlation, and **without** secrets.

## 10. Dependency security

- Python dependencies are pinned in the backend requirements; models run offline in the
  evaluation configuration (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`).
- **Frontend dev dependencies:** `npm audit` reports **7 vulnerabilities (5 moderate,
  1 high, 1 critical)** originating from **development-only** tooling introduced for
  testing (`vitest-axe`, transitive deps of the test/e2e stack). These are **not** part
  of the production browser bundle. They are recorded here honestly and left unpatched
  because the available fixes are breaking major-version bumps of test tooling; they do
  not ship to users. Re-run `npm audit` in `frontend/` to see the current tree.

## 11. What is NOT claimed

- No formal security audit, penetration test, or certification has been performed.
- No authentication/authorization exists yet (Section 8).
- No secret-scanning or SBOM generation is wired into CI.

These are stated plainly to avoid overclaiming; see `docs/limitations.md` and the
completion ledger for status.
