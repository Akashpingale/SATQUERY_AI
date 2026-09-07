# Spec 02 — Frontend / UX · Tasks

- [x] 1. Phase 1 shell (see spec 01): Vite+React+TS+Tailwind+shadcn-style, router,
      QueryClient, API client, Header + SystemStatus + panel placeholders, first test.
      _Reqs: 1, 7_

- [ ] 2. Design system: expand `ui/` primitives (badge, card, dialog, tabs, tooltip,
      skeleton, alert); theme tokens; dark-mode-safe contrast.
      _Reqs: 6_

- [ ] 3. Upload panel: `UploadSlot` (drag-drop + picker) for optical/SAR/T1/T2;
      validation status + metadata preview; replace/remove.
      _Reqs: 2_ · _Depends: uploads API (spec 13), raster ingestion (spec 03)_

- [ ] 4. Visualization: `ImageViewer` zoom/pan/fit; layer toggles; T1/T2 compare;
      overlay boxes/masks/change; legend; coordinates.
      _Reqs: 3_ · _Depends: evidence API (spec 12)_

- [ ] 5. Assistant + results: `QueryBox` (RHF+Zod, examples); `ResultCard`;
      `ConfidenceBadge`; `EvidencePanel`; `ExecutionTrace`.
      _Reqs: 1, 4_ · _Depends: analysis API (spec 13), agent (spec 05)_

- [ ] 6. History + result detail + report download.
      _Reqs: 5_ · _Depends: history/report API (spec 13)_

- [ ] 7. Demo gallery running the real workflow; settings; help/about; not-found.
      _Reqs: 5_

- [ ] 8. Accessibility pass: keyboard nav, focus states, ARIA live regions, alt text,
      axe-core tests; request manual NVDA verification.
      _Reqs: 6_

- [ ] 9. E2E: Playwright critical flows + negative/error states.
      _Reqs: 1–7_

> This spec is delivered incrementally across Phases 2–8; the Phase 1 shell (task 1)
> is complete. Remaining tasks unblock as their backend dependencies land.
