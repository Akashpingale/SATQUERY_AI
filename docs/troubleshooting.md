# Troubleshooting

Practical fixes for issues you may hit while setting up, running, or demoing
SatQuery AI. It is written against the actual configuration and code paths in this
repository. For usage, see `docs/user-guide.md`; for setup, see `docs/setup.md`.

---

## 1. Setup and environment

### `uv`, Python, or venv issues
- Backend requires **Python 3.11+**. Confirm with `python --version`.
- Use the project virtual environment. On Windows the interpreter is
  `.\.venv\Scripts\python.exe`.
- Reinstall backend deps into the venv if imports fail.

### Node / frontend install
- The frontend uses Node + npm in `frontend/`. Run `npm install` there.
- Type errors on first run usually mean deps are not installed yet.

### Docker
- Validate the compose file without starting anything:
  `docker compose --project-directory d:\SatQuery-AI config --quiet`.
- A clean-machine bring-up is described in `docs/setup.md`.

## 2. Models load slowly, fail to download, or error offline

The evaluation configuration runs models **offline**. Set these before running the
backend or tests so nothing reaches out to the network:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HOME = "D:\SatQuery-AI\models\hf-cache"
```

- If a model errors because weights are missing, the cache at `HF_HOME` does not contain
  it. Populate the cache while online once, then run offline.
- Local dev runs on **CPU**. Heavy/GPU inference is separable and optional; the app
  degrades gracefully and respects the `DEVICE` config.
- Loaded models are reused; the first request in a session is the slow one.

## 3. Uploads

### "Unsupported file type" / HTTP 415
The declared content type is not on the allowlist. Upload a GeoTIFF/TIFF (or PNG/JPEG
for benchmark data). Note the content-type check is advisory — the file must also be a
**real, readable raster**.

### Upload rejected even though it has a `.tif` name
Validation opens the bytes as a raster. A file that is not actually a raster is rejected
and its stored bytes are deleted. Re-export a valid GeoTIFF.

### "File exceeds the maximum upload size"
Default limit is **200 MB** (`max_upload_mb` in `backend/app/core/config.py`). Raise it
via environment configuration if your data is legitimately larger, or tile the raster.

### Preview does not appear
Preview rendering is best-effort; a failure is logged server-side but does not fail the
upload. The image is still usable for analysis. Check backend logs for the warning.

## 4. Analysis

### "Incompatible" images / validation message instead of an answer
This is intended behavior — the system does **not** silently analyze incompatible pairs.
Common causes: mismatched CRS, an optical-only pair placed in optical+SAR slots, or a
missing modality. Match the imagery to the slot and retry.

### Confidence is "Low" or "Unable to determine"
Not a bug. Confidence is computed from real signals. Read the "Why this confidence"
explanation and the execution trace to see which signals were weak.

### A rejected request returns an explanation, not a blank
Expected. Unsupported or rejected requests return an explanatory answer describing why.

### Analysis seems to hang
Requests are bounded by `request_timeout_seconds` (default **120s**). The first request
in a session also pays the model-load cost. If it consistently times out, check backend
logs and available memory.

### My "instructions" in the query were ignored
Also intended. The query is untrusted **data**. Instruction-like fragments (e.g. "ignore
previous instructions") are stripped, logged, and reported in the trace as a warning.
They never change tool selection or parameters.

## 5. Evidence and reports

### Evidence image shows a placeholder
The frontend renders a fallback when an evidence image fails to load, instead of a
broken image. Confirm the backend is running and the evidence/preview endpoints are
reachable.

### Evidence or preview returns 404
All file access is confined to the storage root via `resolve_within`. A path that would
escape the root (or a missing artifact) returns 404 by design. Confirm the artifact
exists under the storage root and the stored path is correct.

### Downloaded report shows no images
The report embeds evidence as base64 data URIs at generation time. If images are absent,
the underlying evidence artifacts were missing when the report was generated. Re-run the
analysis so the artifacts exist, then download again.

## 6. Frontend dev server and tests

- **Vitest** unit tests: run `npm run test` in `frontend/`. The config scopes unit tests
  to `tests/**` and excludes `e2e/**`.
- **Playwright** E2E: `npm run e2e` (installs browsers with `npm run e2e:install` first
  if needed). The E2E run starts an **isolated backend** (`scripts/e2e_backend.py`, a
  temp SQLite DB and storage) plus the Vite dev server on `:5173`; it does not touch your
  dev database.
- **On Windows**, Playwright web-server paths must be absolute; forward-slash relative
  paths (`../`) are rejected by `cmd.exe`. This is already handled in
  `frontend/playwright.config.ts`.
- **Accessibility** tests: `npm run test:a11y`. The `color-contrast` axe rule is disabled
  because jsdom cannot compute layout; it is verified manually and documented in
  `docs/accessibility.md`.

## 7. `npm audit` reports vulnerabilities

`npm audit` in `frontend/` reports vulnerabilities originating from **development-only**
test tooling (`vitest-axe` and transitive test/e2e deps). These are not part of the
production browser bundle. See `docs/security.md` §10 for the honest status.

## 8. Database and storage

- PostgreSQL stores **metadata only**; rasters and artifacts live on the filesystem
  under the storage root. Backend tests run on **SQLite** with no running Postgres.
- If you see stale data across runs, confirm which database the process is pointed at
  (dev vs. the isolated E2E temp DB).

## 9. Still stuck?

- Correlate a failing request using the per-request `X-Request-ID` in the response and
  server logs.
- Client errors are sanitized on purpose; the actionable detail is in the **server-side
  logs**, not the client response.
- Check `docs/limitations.md` for known gaps that are expected rather than bugs.
