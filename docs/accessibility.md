# Accessibility

SatQuery AI targets **WCAG 2.1 AA**. This document records what is enforced automatically, what
was checked manually, and what still requires a screen-reader pass. It is honest about the last
point: an automated axe pass is **not** a substitute for a real screen-reader test, and this
file never claims a screen-reader pass that was not performed.

## Status summary

| Check | Status | Evidence |
|---|---|---|
| Automated axe (WCAG rules, jsdom) | **DONE** | `frontend/tests/accessibility.test.tsx` — 13 states, 0 violations |
| Semantic structure, labels, roles, alt text, button names | **DONE** | Enforced by axe + code review |
| Keyboard operability (manual, code-level) | **DONE** | See "Keyboard verification" below |
| Colour contrast | **PARTIAL** | Not computable in jsdom; verified manually against the palette |
| NVDA / screen-reader pass | **NOT YET VERIFIED** | Requires a Windows + NVDA session; procedure documented below |

## Automated testing (A11Y-2)

`frontend/tests/accessibility.test.tsx` runs `axe-core` (via `vitest-axe`) against the important
application states and asserts **zero violations**:

- query panel — empty, running, and error states
- completed / rejected / unavailable / adapted result cards
- evidence panel (cross-modal evidence)
- execution trace, confidence badge
- visualization panel (single image and optical+SAR)
- history panel (empty state)

Run it:

```powershell
cd frontend
npx vitest run tests/accessibility.test.tsx
```

**One rule is excluded, with reason:** `color-contrast` is disabled in these tests because
jsdom does not compute layout or resolve CSS custom properties, so axe cannot evaluate contrast
in that environment. No other rule is disabled. Contrast is instead verified manually (below).
Disabling axe rules to force a pass is not done anywhere else.

## Colour contrast (manual)

The Tailwind theme uses foreground/background pairs that meet AA (≥4.5:1 for body text, ≥3:1 for
large text and UI components):

- Body text `#1a1f24`-family on white/`background` — well above 4.5:1.
- Muted text (`text-muted-foreground`) is used only for secondary/large text, not for essential
  small body copy that lacks another cue.
- **Status is never conveyed by colour alone**: every badge and trace row pairs its colour with
  an icon and a text label (e.g. "Completed" + check icon; "Warning" + triangle icon), and score
  bars print the numeric percentage next to the bar.

If the theme changes, re-verify contrast with a checker (e.g. the browser DevTools contrast
tool) on: primary buttons, badges, trace status text, and muted captions.

## Keyboard verification (manual, code-level)

Verified by inspection of the components; all interactive elements are native or correctly
role-annotated:

- **Focus visibility:** every interactive control uses `focus-visible:ring-2 focus-visible:ring-ring`.
- **Query form:** the textarea has an associated `<label>` (`htmlFor="query"`) and an
  `aria-describedby` help line; Analyze is a real `<button type="submit">`.
- **Example chips:** real `<button>` elements in a labelled list; reachable and activatable by
  keyboard; disabled while a run is in flight.
- **Upload slots, remove, compatibility, reopen-from-history:** native buttons/inputs.
- **Visualization mode switch:** a real `role="tablist"` with `role="tab"` + `aria-selected`.
- **Evidence and trace:** native `<details>/<summary>` disclosures — keyboard-operable by
  default; lists use `<ul>/<ol>` with `aria-label`.
- **Live regions:** the assistant result area is `aria-live="polite"` with `aria-busy`, so a
  running analysis and its result are announced; errors use `role="alert"`.
- **Report download:** a real `<a download href>` link, focusable and activatable.
- **Images:** every `<img>` has meaningful `alt`; decorative icons are `aria-hidden="true"`.

## NVDA / screen-reader validation (A11Y-3) — NOT YET VERIFIED

A real NVDA pass has **not** been executed, because this build/CI environment has no NVDA and no
audio session. It must be run manually on Windows before the submission is called
accessibility-complete. The exact procedure:

1. Install NVDA (free, nvaccess.org) on Windows. Start the app: backend on :8000, frontend on
   :5173 (see `docs/setup.md`).
2. With NVDA running, load `http://localhost:5173` and verify, using keyboard only:
   - **Page/landmarks:** heading and landmark navigation (`H`, `D`) reach Inputs, Visualization,
     Assistant, and History regions.
   - **Upload controls:** each slot's purpose (Optical / SAR / T1 / T2) is announced; the file
     input is reachable and labelled.
   - **Modality/role:** the slot labels and any validation/compatibility status are announced,
     not conveyed by colour alone.
   - **Query input:** the textarea label and help text are read; character limit is perceivable.
   - **Example chips:** each chip's text is announced; activating one fills the query box.
   - **Submit:** the Analyze button name is read; on submit, the polite live region announces
     "Analyzing…" and then the result.
   - **Result:** the "Analysis result" region heading, status ("Completed"/"Rejected"/…),
     answer text, confidence level and signals are all announced.
   - **Evidence & trace:** the `<details>` disclosures expand/collapse and their content
     (labels, measured values, trace steps) is read in order.
   - **Errors:** an error (`role="alert"`) is announced immediately.
   - **Report download:** the "Download report" link is announced and activatable.
3. Record findings (pass/fail per item, NVDA + browser versions, date) in the table below and
   fix any issues found.

### NVDA results

_Not yet run. Fill in after executing the procedure above._

| Area | Result | Notes |
|---|---|---|
| Landmarks / headings | — | |
| Upload controls | — | |
| Query input + chips | — | |
| Submit + live announcement | — | |
| Result / confidence | — | |
| Evidence / trace | — | |
| Errors | — | |
| Report download | — | |

## Known limitations

- The large image viewer supports zoom/pan by toolbar buttons (keyboard-operable); pointer
  pan is mouse/touch only, which is acceptable because the same views are reachable without it.
- Colour-contrast is verified manually, not in CI (jsdom limitation).
- NVDA verification is outstanding (above). Until it is done, accessibility is **PARTIAL**, not
  complete, in `docs/traceability.md` and the completion ledger.
