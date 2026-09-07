---
inclusion: always
---

# AI / ML Rules

## No fakes (critical)
- **Never fabricate** AI outputs, confidence values, model names, execution traces,
  benchmark scores, or dataset provenance.
- Temporary mocks are allowed **only** during isolated frontend development and must
  be clearly labeled as mock and removed before a feature is "done".
- If something cannot be evaluated, report `NOT EVALUATED — dependency unavailable`.

## Genuine remote-sensing adaptation
- A generic VLM with a "remote sensing" prompt is **not** adaptation. Implement a
  real LoRA/QLoRA adaptation route on remote-sensing data (BigEarthNet is a required
  route). The adapted checkpoint must be wired into the real serving workflow.
- Do not finalize the base VLM until hardware, license, VRAM, and feasibility are
  checked. The architecture must allow model replacement via the registry.
- Track for reproducibility: base model, version, dataset+version, preprocessing,
  hyperparameters, seed, adapter/checkpoint, hardware, and eval metrics.

## Genuine optical-SAR fusion
- Do not concatenate two models' text outputs and call it fusion. Implement real
  multimodal integration (feature/embedding/cross-attention/learned fusion) and
  document the chosen method and why it is genuine.

## Agent & inference discipline
- Specialists implement a **uniform input/output contract** and are registered.
- The agent selects only registered tools with **allowlisted, validated parameters**.
- Compute path: **CPU for local dev**; GPU training/heavy inference is separable and
  runs on a cloud GPU. Respect `DEVICE` config; degrade gracefully (small/quantized).
- Manage GPU/CPU memory; apply timeouts; reuse loaded models.
