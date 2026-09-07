---
inclusion: always
---

# Security Rules

Treat **all uploaded images and user queries as untrusted input**.

## Uploads & files
- Enforce extension + **MIME** checks, size limits, and corruption checks before use.
- Generate **safe unique filenames**; never trust client-provided names.
- **Path-traversal protection**: resolve and confine all paths under configured
  storage roots. Reject anything that escapes.
- Isolate temp files; clean them up. Apply resource and time limits.

## Agent & model safety
- Validate all model/tool parameters against **allowlisted schemas**. Reject invalid plans.
- **Never generate or execute arbitrary commands** from user input.
- Prompt-injection resistance: separate user task from system/tool instructions; treat
  text extracted from images as untrusted; restrict available tools; keep secrets out
  of model context; log blocked unsafe tool attempts.

## Secrets
- Secrets come from `.env` / a secret manager only. **Never** hard-code, commit, log,
  or ship them to the frontend. `.env` is gitignored; `.env.example` documents keys.

## Errors & exposure
- Return sanitized errors to clients; log details server-side without secrets.
- Network-exposed endpoints without auth must be explicitly flagged as such.
