# Spec 07 — Remote-Sensing VLM Adaptation · Design (as built)

This is the design that was actually implemented in Phase 5, including the decisions that were
reversed after measurement. Measured numbers live in `docs/training.md`,
`docs/evaluation.md`, and `docs/datasets.md`.

## Shape of the change

```
scripts/verify_phase5_gpu.py     hard gate: real CUDA forward/backward/optimizer/fp16
        |
scripts/inspect_bigearthnet_*.py measure sizes, schema, pairing BEFORE downloading
scripts/download_bigearthnet.py  official annotations + subset imagery
scripts/prepare_bigearthnet_txt.py -> data/bigearthnet/prepared/*.jsonl + SHA256 manifest
        |
ml/datasets/bigearthnet.py       LMDB reader, RGB conversion, split preparation, manifest
ml/training/lora_clip.py         LoRA config, CaptionDataset, collator, InfoNCE, trainer
scripts/phase5_smoke_train.py    13-stage pipeline proof on a handful of samples
scripts/train_phase5.py          real run -> models/phase5/adapter + reports/phase5/*.json
        |
ml/evaluation/retrieval.py       image<->text retrieval, both directions, chance level
scripts/evaluate_phase5.py       baseline vs adapted, identical preprocessing, one process
        |
ml/inference/adapted_clip.py     variant resolution, adapter description, cached loading
backend/app/registry/models.py   vqa_adapted / captioning_adapted entries
backend/app/agent/planner.py     adapted preference + recorded selection reason
backend/app/agent/trace.py       model provenance step
scripts/verify_phase5.py         25 end-to-end checks through the real API
```

Layering is preserved: `backend` depends on `ml.inference` only. `ml.training` is imported
lazily by the adapter loader, so serving does not require the training stack beyond `peft`.

## Key design decisions

### Base model: CLIP + LoRA, not a generative VLM
Decided against Qwen2-VL-2B / PaliGemma-3B on **measured** VRAM (4.00 GB total): they do not
fit even under 4-bit quantisation once activations and optimizer state are included. Decided
against training a generative captioner because small generative models produce fluent,
confidently wrong descriptions of satellite imagery, which conflicts with the project's
no-fabrication rule. CLIP was also the only choice that improves the **existing** product:
the Phase 4 VQA and captioning specialists already read CLIP embeddings.

### Objective: symmetric InfoNCE on captioning rows only
A dual encoder needs matched image–description pairs. The `binary`, `mcq`, and `bounding box`
rows (9.1 M of them) were rejected: training a text tower on `"yes"`/`"no"` or coordinate
strings supplies no scene description. LoRA targets `q_proj,k_proj,v_proj,out_proj`, which
exist in **both** towers, so image and text representations adapt together.

### Data: official annotations + unofficial imagery mirror
The official Zenodo imagery was measured at **109.88 GB** compressed (~170–220 GB extracted)
against 13.54 GB free on C: and 258.17 GB on D: — infeasible within the phase. A 2.31 GB
Lithuania/Summer LMDB mirror was used for pixels, while **captions and splits come from the
official annotation file**. Mitigation: every patch id is validated against the official
annotations (0 annotation-side mismatches, 95.30 % mirror coverage), and the caveat is
repeated in the manifest, the module docstring, and `docs/datasets.md`.

### Splits: the dataset's own `split` column
Custom splitting was rejected because BigEarthNet patches are spatially adjacent and a naive
shuffle risks leakage between train and test.

### Registry: separate entries, not a runtime `variant` parameter
`vqa_adapted` / `captioning_adapted` are distinct registry entries. A `variant` parameter on
the existing tools was rejected: separate entries make the plan, the trace, and the UI state
unambiguously which model ran, and they keep the parameter allowlist simple.

### Grounding intentionally excluded
No grounding-specific training or evaluation was performed, so `ADAPTED_PREFERENCE` has no
`grounding` key and a test asserts it stays that way. Claiming a grounding improvement from a
retrieval result would be unfounded.

### Status stays `degraded`
Real inference, real improvement, but uncalibrated scores measured on one country and one
season with no task-accuracy benchmark. `available` requires Phase 9 evidence.

## Bugs found by measurement (and their fixes)

| Problem | Root cause | Fix |
|---|---|---|
| `from transformers import CLIPModel` aborted the interpreter | transformers 5.16.1 incompatible with this torch build | Pin `transformers>=4.44,<5` in `pyproject.toml` and the Dockerfile |
| `TypeError: CLIPModel.forward() got an unexpected keyword argument 'inputs_embeds'` | PEFT `task_type="FEATURE_EXTRACTION"` injects args a dual encoder does not accept | Omit `task_type`; use the generic `PeftModel` passthrough |
| `lmdb.Error: environment ... already open in this process` | Multiple readers (train/val/eval) opening the same LMDB | `_ENV_CACHE` keyed by `(pid, path)` + `close_shared_envs()`; pid in the key so forked workers open their own handle |
| Training images were near-black (mean 24/255) | Fixed reflectance ceiling of 3000 vs measured p50=277, p99=528, max=1148 | Per-image joint 2–98 percentile stretch, matching `geospatial.render` at inference time |
| Trace lost the model-selection reason | `validate_plan` dropped `selection_notes` | Preserve `selection_notes` through validation |
| Docstring claimed ~11 % of captions truncated | Never measured | Measured with the real tokenizer: **84.0 %** truncated, mean 49.1 % of tokens kept (`scripts/measure_caption_tokens.py`); docstring and docs corrected |
| `tsc -b` emitted `frontend/vite.config.js`, which Vite resolves **before** `vite.config.ts` | `tsconfig.node.json` had `composite` with no `outDir` | Redirect emit + build info into `node_modules/.tmp`; delete the stale file; gitignore `*.tsbuildinfo` |

## Safety and honesty mechanisms

- Training **raises** if CUDA is unavailable rather than silently running on CPU.
- The smoke test asserts the adapter **changes** model output (max logit delta 0.672132), so a
  saved no-op cannot pass as a trained adapter.
- Adapter absence produces a baseline fallback **with the reason recorded in the trace**.
- Evaluation reports the **chance level** next to every metric so results cannot be over-read.
- Tests run with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`; model-dependent tests skip
  rather than download or assert against invented outputs.
- Every script writes its own report file with `flush()` + `os.fsync()`, because this shell
  truncates stdout from long-running processes; the file is the record, not the scrollback.

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| `PHASE5_ADAPTER_DIR` | `models/phase5/adapter` | Adapter location; absolute required in Docker |
| `HF_HOME` | `./models/hf-cache` | Weight/dataset cache; point at a drive with space |
| `DEVICE` | `cpu` | Serving device; training forces CUDA |

## Not in scope for this spec
Optical–SAR fusion (spec 11, Phase 7) — Sentinel-1 is recorded in the manifest and unused.
Temporal change (spec 10, Phase 6). Benchmark accuracy and calibration (spec 14, Phase 9).
