# User Guide

This guide explains how to use SatQuery AI end to end: uploading imagery, asking a
question, and reading the answer, evidence, confidence, and execution trace. It
describes the application as it actually behaves. For installation and running the
stack, see `docs/setup.md`. For a scripted demo walkthrough, see `docs/demo.md`.

---

## 1. What SatQuery AI does

SatQuery AI is an **agentic remote-sensing analysis assistant**. You upload satellite
imagery and ask a natural-language question. The system understands the question,
validates the imagery, selects the right specialist model/tool from its registry, runs
real analysis, and returns:

- an **answer**,
- **visual evidence** (previews and overlays),
- a **confidence level** (High / Medium / Low / Unable to determine), and
- a concise, auditable **execution trace**.

It is **not** a generic chatbot. Every capability runs the real production workflow;
there are no canned answers.

## 2. The workspace layout

The main screen (`Workspace`) is a three-zone geospatial workbench, each zone a titled
panel:

- **Input workspace** (left) — modality-grouped upload cards (Optical / SAR, a Temporal
  pair, and an Optional / advanced single-image slot), an **analysis configuration**
  summary that shows exactly what the agent will analyze, and the History panel below it.
- **Visualization** (centre) — a polished empty state before any upload, then the raster
  preview with zoom/pan and a metadata overlay (dimensions, CRS, resolution). A segmented
  layer control appears when there is more than one view (Single image, Compare T1 / T2,
  Change evidence, Optical + SAR).
- **AI Assistant** (right) — the query console where you type a question, pick from
  grouped **suggested questions**, watch the **analysis pipeline**, and read the result
  (answer, confidence, agent plan, evidence, execution trace, report).

The header shows live system status derived from the real backend: the compute device,
the count of available registry tools, and an online/offline/checking indicator.

## 3. Uploading imagery

1. In the Input panel, choose the slot that matches your data:
   - **Single image** — one optical/multispectral or SAR raster for VQA, captioning, or
     grounding.
   - **Optical + SAR** — two co-registered rasters for optical-SAR fusion.
   - **T1 + T2** — two images of the same area at different times for change analysis.
2. Select a GeoTIFF/TIFF file (PNG/JPEG are also accepted, mainly for benchmark data).
3. The file is validated on the server. If it is not a readable raster, the upload is
   rejected with a clear message and nothing is stored. Files above the size limit
   (default 200 MB) are rejected during upload.
4. A preview appears in the Visualization panel once ingestion succeeds. The most recent
   valid upload is shown automatically.

**Input validation matters:** the system will not silently analyze incompatible pairs
(for example, mismatched CRS or an optical-only pair sent as optical+SAR). It tells you
what is wrong instead of guessing.

## 4. Asking a question

1. Type your question in the AI Assistant console, or click one of the **suggested
   questions**. The suggestions **adapt to what you have uploaded** — a temporal pair
   surfaces change questions, an optical+SAR pair surfaces fusion questions, a single
   image surfaces VQA/captioning/grounding questions.
2. Submit the question. The UI collects the relevant asset ids automatically:
   - a T1+T2 pair takes precedence for change tasks,
   - then optical+SAR for fusion,
   - otherwise the single available image.
   The **backend** decides which task the combination supports — the UI never hard-codes
   routing.
3. While the analysis runs, the console shows the **analysis pipeline** (understand →
   validate → select tool → analyze → build evidence → confidence) as an in-progress
   state. Once the result arrives, the same six stages are shown as an **agent plan**
   whose per-stage status is derived from the real execution trace.

Your question is treated as untrusted data. Instruction-like text (attempts to change
the system's behavior) is neutralized and noted in the trace; it never changes which
tools run.

## 5. Reading the result

Every result exposes four things:

- **Answer** — the specialist's response to your question. A rejected or unsupported
  request still returns an explanatory answer describing why, not a blank.
- **Confidence** — one of High / Medium / Low / Unable to determine, with a
  "Why this confidence" explanation of the signals behind it. Confidence is computed
  from real signals; it is never invented.
- **Evidence** — image previews and overlays (for example, grounding boxes or change
  regions) shown in the Visualization panel. If an evidence image fails to load, a
  fallback placeholder is shown instead of a broken image.
- **Execution trace** — an ordered, human-readable list of the steps the system took
  (understanding, validation, tool selection, execution, evidence, confidence). The
  trace is auditable and **never exposes model chain-of-thought**.

## 6. Downloading a report

Each completed analysis can be exported as a **self-contained HTML report** via the
**"Download report"** link on the result card. The report:

- includes the question, answer, confidence and its rationale, the model id/version
  used, the execution trace, and warnings,
- **embeds the evidence images inline** (as base64 data URIs), so the file is fully
  portable — no server connection needed to view it later,
- can be opened in any browser and printed to PDF if you need a PDF copy.

Reports are generated from the persisted analysis record, so you can download a report
for a past analysis reopened from history.

## 7. Using history

The History panel lists previous analyses with loading, empty, and error states handled
explicitly. Selecting an entry reopens that analysis — its answer, confidence, evidence,
and trace — and lets you download its report.

## 8. Supported inputs at a glance

| Input | Example task |
| --- | --- |
| Single optical/multispectral image | VQA, captioning, grounding |
| Single SAR image | VQA / captioning |
| Optical + SAR (co-registered) | Optical-SAR fusion |
| T1 + T2 (same area, two dates) | Change description, change-VQA |

Formats: GeoTIFF/TIFF (primary), plus PNG/JPEG for benchmark data.

## 9. When something goes wrong

- **Upload rejected** — the file is not a valid raster, uses an unsupported format, or
  exceeds the size limit. Try a valid GeoTIFF.
- **"Incompatible" or validation message** — the two images cannot be analyzed together
  (e.g., mismatched CRS or wrong modality pairing). Check that your pair matches the slot.
- **Low / Unable to determine confidence** — the system is telling you honestly that the
  evidence is weak; read the "Why this confidence" note and the trace.

For deeper diagnostics and developer issues, see `docs/troubleshooting.md`.
