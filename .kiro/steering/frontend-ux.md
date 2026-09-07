---
inclusion: fileMatch
fileMatchPattern: 'frontend/**'
---

# Frontend / UX Rules

The frontend is a **core product requirement**: modern, clean, professional,
responsive, accessible, and beginner-friendly. Judges must find it obvious to use.

## Stack
React + TypeScript + Vite + Tailwind CSS + shadcn/ui + React Router + TanStack Query
+ React Hook Form + Zod + Lucide icons. Tests: Vitest + React Testing Library;
E2E: Playwright.

## The workflow must be visually obvious
```
Upload image -> Ask question -> Agent understands -> Agent selects tools/models
-> Analysis -> Answer -> Visual evidence -> Confidence -> Execution trace
```

## Layout
- Header (branding, system status, help/about, settings).
- Input panel (drag-drop optical/SAR/T1/T2, format indicators, validation status,
  metadata preview, replace/remove).
- Visualization (large viewer: zoom/pan/fit, layer toggles, optical/SAR switch,
  T1/T2 compare, boxes/masks/overlays, legend, coordinates when available).
- Assistant panel (query box, example questions, Analyze, answer, confidence,
  evidence, warnings, selected task, selected model/tool, execution summary).
- History (past analyses: timestamp, task, status, reopen).

## Rules
- Design-system components; reuse, don't duplicate. Use the shared API client only.
- Always handle **empty / loading / error** states. No raw stack traces in the UI.
- Explain unsupported formats and incompatible pairs in plain language.
- **Accessibility is mandatory**: semantic HTML, keyboard navigation, visible focus,
  labels, screen-reader support, sufficient contrast, **status never by color alone**,
  meaningful alt text, accessible errors.
- Avoid unnecessary animation; keep important evidence visible.
- Mock data is allowed only in isolated component dev and must be clearly labeled.
