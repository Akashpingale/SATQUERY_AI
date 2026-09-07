# Spec 02 — Frontend / UX · Requirements

## Introduction
The frontend is a core product requirement: a modern, clean, professional,
responsive, accessible web app that makes the remote-sensing analysis workflow
obvious and judge-friendly. This spec covers the full UX beyond the Phase 1 shell:
upload, visualization, the assistant/query experience, results (answer + evidence +
confidence + trace), history, reports, demo mode, settings, and help — with strong
accessibility. It consumes the backend API contracts only; it never depends on a
specific model.

## Requirements

### Requirement 1 — Guided analysis workflow
**User story:** As a first-time user, I want the workflow to be visually obvious, so
that I can go from image to answer without instructions.

**Acceptance criteria**
1. WHEN the workspace loads THEN the UI SHALL present the flow: upload → ask →
   analyze → answer → evidence → confidence → trace.
2. WHEN no image is uploaded THEN the assistant SHALL guide the user to upload first.
3. WHEN an analysis is running THEN the UI SHALL show a clear loading state and the
   selected task/model once known.

### Requirement 2 — Upload & input panel
**User story:** As a user, I want to upload optical, SAR, and temporal images easily,
so that I can analyze different configurations.

**Acceptance criteria**
1. WHEN uploading THEN the panel SHALL support drag-and-drop and file picker for
   optical, SAR, T1, and T2 slots.
2. WHEN a file is added THEN the UI SHALL show supported-format indicators, a
   validation status, and a metadata preview (dimensions, bands, CRS when available).
3. IF a file is unsupported or invalid THEN the UI SHALL explain why in plain language.
4. WHEN a file is added THEN the user SHALL be able to replace or remove it.

### Requirement 3 — Visualization
**User story:** As an analyst, I want to inspect imagery and evidence visually.

**Acceptance criteria**
1. WHEN an image is loaded THEN the viewer SHALL support zoom, pan, and fit-to-screen.
2. WHEN multiple layers exist THEN the UI SHALL toggle optical/SAR and compare T1/T2.
3. WHEN evidence exists THEN the viewer SHALL overlay boxes/masks/highlights/change
   overlays with a legend, and show coordinates when georeferencing is available.

### Requirement 4 — Assistant & results
**User story:** As a user, I want to ask a natural-language question and get a clear,
trustworthy answer.

**Acceptance criteria**
1. WHEN I submit a query THEN the assistant SHALL display the answer, confidence
   level, evidence, warnings, the selected task, and the selected model/tool.
2. WHEN results return THEN a concise execution trace SHALL be viewable (no
   chain-of-thought).
3. WHEN example questions are offered THEN selecting one SHALL populate the query box.

### Requirement 5 — History, reports, demo, settings, help
**Acceptance criteria**
1. WHEN analyses complete THEN history SHALL list them (timestamp, task, status) and
   allow reopening.
2. WHEN a result exists THEN the user SHALL be able to download a report.
3. WHEN demo mode is opened THEN curated cases SHALL run the real workflow.
4. WHEN settings/help are opened THEN the app SHALL show configuration and guidance.

### Requirement 6 — Accessibility (mandatory)
**Acceptance criteria**
1. WHEN navigating THEN all interactive elements SHALL be keyboard reachable with a
   visible focus indicator.
2. WHEN using a screen reader THEN semantic HTML, labels, and alt text SHALL convey
   state and content.
3. WHEN status/severity is shown THEN it SHALL NOT rely on color alone.
4. WHEN measured THEN contrast SHALL meet WCAG AA where feasible; automated axe checks
   SHALL pass in tests. (Full conformance requires manual screen-reader testing.)

### Requirement 7 — Robust states & errors
**Acceptance criteria**
1. WHEN any async operation is pending/empty/failed THEN a dedicated state SHALL show.
2. WHEN the backend errors THEN the UI SHALL show a friendly message, never a raw
   stack trace.

## Dependencies
- Phase 1 frontend shell + API client.
- Backend endpoints: uploads, analysis, analysis/{id}, trace, evidence, report, history
  (delivered across Phases 3–8).

## User inputs required
- Optional: brand assets/colors. A clean neutral default is used otherwise.
- Human NVDA/screen-reader verification (requested at finalization).

## Known risks
- Large raster previews in-browser → rely on backend-provided downsampled previews/tiles.
- Accessibility conformance needs manual testing, not just automated checks.
