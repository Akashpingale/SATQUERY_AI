# Spec 07 — Remote-Sensing VLM Adaptation

> Phase 5. **Implemented.** Requirements below are the acceptance contract; `design.md`
> records what was actually built and `tasks.md` tracks completion. Measured results live in
> `docs/training.md`, `docs/evaluation.md`, `docs/datasets.md`, and `reports/phase5/*.json`.

## Introduction
A genuine remote-sensing adaptation of an open-source VLM (LoRA/QLoRA), with a BigEarthNet
training route, evaluated and integrated into the real serving workflow. A generic VLM plus a
"remote sensing" prompt does NOT satisfy this requirement.

## Requirements

### R7.1 — Model selection must follow a hardware/licence feasibility check
The base model is chosen **after** measuring real VRAM, licence terms, and integration value —
not pre-locked. The decision and the rejected alternatives must be recorded.

**Met.** GPU measured first (`scripts/verify_phase5_gpu.py`: RTX 2050, 4.00 GB, CC 8.6,
CUDA 12.4). `flax-community/clip-rsicd-v2` selected; 2–3 B generative VLMs rejected on
measured VRAM. Rationale and rejections in `docs/training.md`.

### R7.2 — Real dataset pipeline with recorded provenance
Acquisition, validation, preprocessing, and splits for a BigEarthNet route, with licence and
provenance recorded and exclusions counted rather than silently dropped.

**Met.** `ml/datasets/bigearthnet.py` + `scripts/{inspect,download,prepare}_*`. 8,362 samples
retained, 0 excluded, official `split` column used, SHA256-pinned manifest at
`data/manifests/bigearthnet_txt_manifest.json`. Provenance — including that the **imagery came
from an unofficial mirror** — documented in `docs/datasets.md`.

### R7.3 — Real LoRA/QLoRA training with full reproducibility metadata
Record base model, dataset + version, preprocessing, hyperparameters, seed, adapter, hardware,
and metrics. No fabricated training results.

**Met.** `ml/training/lora_clip.py`, `scripts/train_phase5.py`. Real GPU run: 6 epochs,
750 optimizer steps, 7.6 min, peak 2,219.2 MB VRAM, train loss 2.4049 → 1.2647, validation
2.5688 → 2.3774. Metadata in `reports/phase5/{training_config,training_metrics,environment}.json`.
**Known gap:** `environment.json` records `git_commit: "HEAD"` because the repository has no
commits yet.

### R7.4 — Honest evaluation with real metrics
Evaluate the adapted model against the unadapted baseline on data not used for training.
Report `NOT EVALUATED` where a dependency is unavailable.

**Met.** `ml/evaluation/retrieval.py`, `scripts/evaluate_phase5.py`. Held-out `test` split
(n=2,053), identical preprocessing for both models, chance level reported alongside results:
mean recall 0.0115 → 0.0791, median rank 518 → 85 (i2t) and 645 → 84 (t2i). Everything not
measured is marked `NOT EVALUATED` in `docs/evaluation.md`.

### R7.5 — Adapter served by the real workflow
The adapted checkpoint must be reachable through the registry and used by the live API, not
kept as an offline artifact.

**Met.** `ml/inference/adapted_clip.py` + registry entries `vqa_adapted`, `captioning_adapted`.
Planner selects them with a recorded reason; trace carries base model, adapter path, and
training dataset; evidence carries the adapted version string.
`scripts/verify_phase5.py` proves it through a real upload → analysis request (25/25 checks).

### R7.6 — No silent downgrade, no unearned claims
If the adapter is missing or fails to load, the system serves the baseline and says so. No
capability may claim improvement it did not measure.

**Met.** `resolve_variant()` returns the baseline with a reason recorded in the trace.
Grounding was not adapted, has no adapted registry entry, and a test asserts
`ADAPTED_PREFERENCE` contains no `grounding` key. Registry status stays `degraded`.

### R7.7 — Training separable from serving
Training must not be required to run the product, and must be movable to another GPU host.

**Met.** Backend imports only the adapter loader. Torch/peft are optional extras
(`pip install -e ".[training]"` vs `".[inference]"`). Adapter located by
`PHASE5_ADAPTER_DIR`.

## Dependencies
PyTorch (CUDA build for training), Transformers (`<5` — see design), PEFT, Accelerate,
safetensors, pyarrow, lmdb. Registry (06), inference/specialists (08, 09).

## Acceptance criteria
- [x] Reproducible training run producing an adapter used by the live VQA/captioning path.
- [x] Baseline vs adapted measured on data never used for training.
- [x] Honest metrics recorded; `NOT EVALUATED` used where a dependency is unavailable.
- [x] Provenance and licensing documented, including the unofficial-mirror caveat.
- [x] Offline-safe tests; Phase 4 regression intact.

## Tests
`backend/tests/test_phase5_adaptation.py` — adapter description contract, variant resolution
and fallback, registry separation, planner preference, trace provenance, grounding-not-adapted
assertion, and report-file contracts. Model-dependent tests skip via `models_ready` /
`adapter_present` / `training_reports` fixtures; the suite runs fully offline.

## Resolved risks
- ~~No local GPU; must use a cloud GPU.~~ A real 4 GB CUDA GPU was available and verified.
- ~~Dataset size/licensing.~~ Official imagery measured at 109.88 GB and infeasible on this
  disk; a 2.31 GB CDLA-Permissive subset mirror was used, with the caveat documented.
- VRAM feasibility handled by model choice plus OOM batch-size probing.

## Outstanding (later phases)
- Benchmark task accuracy (RSVQA/LEVIR-CC) and confidence calibration — Phase 9.
- Multi-country/multi-season adaptation and longer text context — not scheduled.
- Registry status stays `degraded` until task accuracy is established.
