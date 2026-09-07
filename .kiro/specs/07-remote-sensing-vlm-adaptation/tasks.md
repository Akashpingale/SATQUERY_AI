# Spec 07 — Remote-Sensing VLM Adaptation · Tasks

- [x] 1. Install CUDA torch and verify **real** GPU execution as a hard gate — CUDA matmul,
      forward/backward/optimizer step, fp16 autocast (`scripts/verify_phase5_gpu.py`;
      RTX 2050, 4.00 GB, CC 8.6, CUDA 12.4)
- [x] 2. Inspect BigEarthNet.txt: real schema (13 cols), 9,553,962 rows, type/split/country
      distributions, licence, and image↔text pairing (`scripts/inspect_bigearthnet_txt.py`)
- [x] 3. Produce a download plan from **measured** source sizes and free disk, then download
      only what is required (109.88 GB official rejected; 2.75 GB downloaded)
- [x] 4. Dataset preparation pipeline + reproducible SHA256-pinned manifest, official splits,
      recorded exclusions (8,362 retained / 0 excluded)
- [x] 5. Select the adaptation model against real VRAM/licence/Phase-4 compatibility; document
      the decision and the rejected alternatives
- [x] 6. Implement the LoRA pipeline: config, LMDB-backed dataset, collator, symmetric InfoNCE
      loss, cosine schedule with warmup, fp16 + GradScaler, trainer
- [x] 7. Tiny GPU smoke train — 13 stages including adapter save, reload, and proof that the
      adapter **changes** output (`scripts/phase5_smoke_train.py`)
- [x] 8. Real training run on the prepared subset; save measured metrics, config, environment
      (6 epochs, 750 steps, 7.6 min, peak 2,219.2 MB VRAM, loss 2.4049 → 1.2647)
- [x] 9. Verify adapter reload in a **fresh process** and run real adapted inference
- [x] 10. Baseline vs adapted on the held-out `test` split with identical preprocessing and
      chance level reported (mean recall 0.0115 → 0.0791; median rank 518 → 85 / 645 → 84)
- [x] 11. Integrate into registry (`vqa_adapted`, `captioning_adapted`), planner preference
      with recorded reason, trace provenance, evidence version string, frontend rendering
- [x] 12. Offline-safe tests + skip fixtures (`models_ready`, `adapter_present`,
      `training_reports`); Phase 4 regression intact
- [x] 13. Full verification (backend, frontend, lint, build, docker config) + docs
      (`datasets.md`, `training.md`, `evaluation.md`, `models.md`, `traceability.md`, README)

## Corrections made during the phase
- [x] Pin `transformers<5` — 5.16.1 aborts the interpreter on `import CLIPModel`
- [x] Drop PEFT `task_type` — its wrappers pass `inputs_embeds` to `CLIPModel.forward()`
- [x] Share LMDB environments per `(pid, path)` — LMDB refuses a second open in one process
- [x] Replace the fixed reflectance ceiling with a per-image 2–98 percentile stretch after
      measuring near-black output (mean 24/255)
- [x] Preserve `selection_notes` through `validate_plan` so the trace keeps the reason
- [x] Correct the caption-truncation figure from an unmeasured "~11 %" to a measured **84.0 %**
      and add `scripts/measure_caption_tokens.py`
- [x] Stop `tsc -b` emitting `frontend/vite.config.js`, which shadowed `vite.config.ts`
- [x] Gitignore the 2.8 GB dataset working copy; track the manifest and `reports/phase5/*.json`
- [x] Install `peft` in the backend image and set `PHASE5_ADAPTER_DIR` in compose, so the
      container can actually serve the adapted variant

## Deliberately not done
- [ ] Grounding adaptation — no grounding training/eval performed, so no gain is claimed
      (a test asserts grounding has no adapted preference)
- [ ] Generative captioning — rejected: fluent-but-wrong output conflicts with the
      no-fabrication rule
- [ ] Optical–SAR fusion — Sentinel-1 recorded but unused; spec 11, Phase 7

## Remaining (later phases)
- [ ] Benchmark task accuracy (RSVQA / LEVIR-CC) and flip registry status from `degraded`
      once accuracy is established — Phase 9
- [ ] Confidence calibration against labelled data — Phase 9
- [ ] Multi-country / multi-season adaptation; longer text context to lift the 84 % caption
      truncation ceiling — not scheduled
- [ ] Pin `environment.json` `git_commit` to a real SHA (currently `"HEAD"`: no commits yet)
