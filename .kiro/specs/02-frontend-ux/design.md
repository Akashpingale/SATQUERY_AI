# Spec 02 — Frontend / UX · Design

## Approach
Component-driven React + TypeScript app with a small design system built on Tailwind
and shadcn/ui-style primitives. Server state via TanStack Query against a single typed
API client. Forms validated with React Hook Form + Zod. Routing via React Router.

## Information architecture / routes
```
/                Home / dashboard (intro + demo entry + system status)
/workspace       Analysis workspace (upload + viewer + assistant + trace)
/history         Past analyses (list + reopen)
/history/:id     Result detail (answer, evidence, confidence, trace, report)
/demo            Curated demo cases (real workflow)
/settings        Configuration
/help            Help / about
*                Accessible not-found / fallback
```

## Layout composition (workspace)
```
┌───────────────────────────────────────────────┐
│ Header: brand · SystemStatus · Help · Settings │
├──────────────┬───────────────────┬─────────────┤
│ Input panel  │ Visualization     │ Assistant   │
│ optical/SAR  │ viewer/overlays   │ query+answer│
│ T1/T2        │ layers/legend     │ confidence  │
│ validation   │ coords            │ evidence    │
│ metadata     │                   │ trace       │
├──────────────┴───────────────────┴─────────────┤
│ History strip (recent analyses)                 │
└───────────────────────────────────────────────┘
```
Responsive: three columns on desktop, stacked with tabs on small screens.

## Key components
- `ui/` primitives: button, badge, card, dialog, tabs, tooltip, skeleton, alert.
- `SystemStatus`: icon + text (Online/Offline/Checking) — never color-only.
- `UploadSlot`: drag-drop + picker, per-slot validation + metadata preview.
- `ImageViewer`: zoom/pan/fit; overlay layer for boxes/masks/change maps; legend.
- `QueryBox`: RHF + Zod; example-question chips.
- `ResultCard`: answer + `ConfidenceBadge` (level + basis) + warnings + selected task/model.
- `EvidencePanel`: visual + machine-readable evidence links.
- `ExecutionTrace`: ordered, concise steps (task, inputs, validation, tool, params
  summary, evidence, confidence) — no chain-of-thought.
- `HistoryList`, `ReportDownload`, `DemoGallery`.

## State & data
- TanStack Query keys per resource; mutations for upload/analysis; optimistic where safe.
- Typed API client wraps `VITE_API_BASE_URL`; central error normalization → friendly UI.
- No secrets in the bundle; only `VITE_`-prefixed config.

## Accessibility design
- Semantic landmarks (`header`, `main`, `nav`, `aside`), labeled controls, focus-visible
  styles, ARIA live regions for async results, alt text for imagery/evidence.
- Status conveyed by icon + text + shape, not hue alone.
- axe-core assertions in component tests; manual NVDA pass tracked at finalization.

## Testing
- Vitest + RTL for components/hooks and loading/empty/error states.
- Playwright E2E: upload → query → analyze → result → evidence → trace → history →
  report → error handling.
