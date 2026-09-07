# Phase 13 — Security / Release Review

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07
**Scope:** lightweight local application security review (no destructive testing, no external attacks). Evidence from source inspection and static scans of `backend/`, Docker config, and `.env.example`.

## Headline

| Readiness | Verdict |
|---|---|
| **LOCAL SIH DEMO READY** | **YES** |
| **PUBLIC INTERNET DEPLOYMENT READY** | **NO** — API is unauthenticated (by design for the demo); add auth + TLS/rate limiting first |

The distinction is intentional and documented in code and `docs/security.md`. The unauthenticated API is acceptable for a local, single-user demo but **must not** be exposed publicly as-is.

## Findings

| Area | Status | Evidence |
|---|---|---|
| Hardcoded credentials / API keys / secrets | **PASS** | Static scan of `backend/**` for `password=`, `secret=`, `api_key=`, tokens → none (only PyTorch `.eval()` matched). Secrets come from env/`.env` only |
| `.env` hygiene | **PASS** | No `.env` in repo (only `.env.example`); `.env` is gitignored. `POSTGRES_PASSWORD` default is a `change_me…` placeholder; `HF_TOKEN` blank |
| Secrets in frontend bundle | **PASS** | Only `VITE_`-prefixed public config (`VITE_API_BASE_URL`); no secrets shipped |
| Debug mode / verbose errors | **PASS** | No `debug=True`; API returns sanitized `HTTPException` details; structured logging server-side |
| Arbitrary command / code execution | **PASS** | No `subprocess`, `os.system`, `Popen`, `eval(`, or `exec(` in `backend/app/**` |
| File upload — type restriction | **PASS** | Extension allowlist (`SUPPORTED_EXTENSIONS`) + advisory MIME allowlist (`is_allowed_content_type`) + **authoritative** raster-open validation; unsupported → 415/422 |
| File upload — size / resource limits | **PASS** | `MAX_UPLOAD_MB=200` enforced (`UploadTooLargeError`→413); `REQUEST_TIMEOUT_SECONDS=120` |
| Filename safety | **PASS** | `sanitize_filename` strips directory parts, NFKD-normalizes, removes null bytes, collapses unsafe chars, bounds length; storage names are uuid-prefixed (client name never trusted) |
| Path traversal | **PASS** | `resolve_within` resolves and confines every path under the storage root; escapes raise `UnsafePathError`. Evidence images served only via hex-token regex confined to the evidence dir |
| Prompt injection / untrusted query | **PASS** | `sanitize_query` length-bounds, strips control chars, neutralizes instruction-like text; tools/params come only from the registry allowlist (agent cannot invent tools/params); text from images treated as untrusted |
| Agent/tool parameter validation | **PASS** | `validate_plan` / `validate_parameters` enforce allowlisted schemas; invalid plans rejected before execution (Flow E) |
| CORS | **PARTIAL** | Restricted to `http://localhost:8080` in the Docker/production compose; dev default lists localhost origins. Tighten to the real origin for any deployment |
| Authentication / authorization | **KNOWN GAP (by design)** | `/uploads` and `/analysis` are unauthenticated; explicitly flagged in route docstrings and `docs/security.md`. **Required before public exposure** |
| Sensitive-data leakage in errors/logs | **PASS** | Client errors sanitized; no stack traces to clients; no secrets logged |
| Network exposure of unauthenticated endpoints | **FLAGGED** | Must be gated (auth) + fronted by TLS/rate limiting before public deployment |

## Static scan commands (reproducible)

```powershell
# secrets / dangerous calls
grep -r "password=|secret=|api_key=|debug=True|shell=True|eval\(|exec\(" backend/**/*.py   # → none (only .eval())
grep -r "subprocess|os.system|Popen" backend/app/**/*.py                                    # → none
# env hygiene
Test-Path .env    # False (only .env.example)
```

## Required controls before PUBLIC deployment (not blocking the local demo)

1. **Authentication + authorization** on `/uploads` and `/analysis` (API key / JWT / session).
2. **TLS** termination and a real origin in `CORS_ORIGINS` (drop localhost).
3. **Rate limiting / quotas** on upload and analysis (CPU-bound model inference is a DoS surface).
4. Set a strong `POSTGRES_PASSWORD` (not the placeholder) and manage via a secret manager.
5. Consider per-user storage isolation and retention/cleanup policies for uploaded rasters.

## Conclusion

No P0 security defects for a local SIH demo: no secrets, no injection/traversal, validated uploads, sanitized queries, allowlisted agent parameters. The **only** release-gating item is authentication (plus TLS/rate-limiting/CORS tightening) before any public/internet deployment — tracked as **P1** in the gap analysis.
