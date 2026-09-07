---
inclusion: always
---

# Product Rules — SatQuery AI (SIH26167)

Source of truth: `SatQuery_AI_Kiro_Master_Plan.md`. These rules are stable; do not
weaken a mandatory requirement without asking the user first.

## What this product is
An **agentic remote-sensing analysis assistant**. The user uploads satellite
imagery and asks a natural-language question; the system understands it, validates
inputs, selects specialist models/tools, runs real analysis, and returns an answer
with **visual evidence, a confidence level, and a concise execution trace**.

## Hard product rules
- It is **not a generic chatbot**. Every mandatory capability is real, executed,
  integrated, and tested — never a canned text reply.
- Mandatory capabilities: single-image **VQA**; at least one more single-image task
  (**grounding** and/or **captioning**); **bi-temporal change** (description and/or
  change-VQA); genuine **optical-SAR fusion**; **remote-sensing VLM adaptation**.
- Every analysis result must expose: **answer + evidence + confidence + execution trace**.
- Support optical/multispectral, SAR, temporal pairs, and co-registered optical+SAR;
  formats GeoTIFF/TIFF (+ PNG/JPEG for benchmark data).
- Validate inputs before analyzing. Never silently analyze incompatible image pairs.
- Confidence levels are **High / Medium / Low / Unable to determine** — never invented.
- Execution trace is observable and auditable; it must **never expose chain-of-thought**.

## Judge / demo readiness
- The six demo flows (VQA, grounding/captioning, temporal change, change-VQA,
  optical-SAR, agent routing) must run on the **real production workflow**.
- Design for a hidden ISRO/SAC set: no hard-coded answers, coordinates, or demo-only
  routing. Assume unseen locations, sensors, and resolutions.

## Definition of done (feature level)
A feature is done only when it is implemented, wired into the API + UI, covered by
tests, and reflected in `docs/traceability.md`. "Code exists" is not "done".
