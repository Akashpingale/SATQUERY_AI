# Phase 13 — VRSBench Grounding: BLOCKED (provisioning attempted)

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07
**Status:** `BLOCKED` — provisioning was **attempted this phase** and failed on measured network throughput, not on missing permission or disk.

## What changed vs Phase 12

Phase 12 documented VRSBench grounding as BLOCKED because the validation images were not present. In Phase 13 the official source was reachable and provisioning was authorized, so a real download was attempted. It is **infeasible in this environment** because of the measured transfer rate. No IoU is reported.

## Provisioning attempt (evidence)

- **Official source reachable:** `https://huggingface.co/datasets/xiang709/VRSBench` → HTTP 200.
- **Exact inventory (HF API):**
  | File | Size |
  |---|---|
  | `Images_train.zip` | 7,972 MB |
  | **`Images_val.zip`** | **3,792 MB** (the needed asset) |
  | `VRSBench_EVAL_referring.json` | 9.8 MB |
  | `Annotations_val.zip` | 12.2 MB |
- **Annotations already official:** the local `D:\dataset\vrsbench\referring.json` (9.8 MB) **matches** the repo's `VRSBench_EVAL_referring.json` (9.8 MB) — 16,159 referring expressions over 9,318 images. Correspondence confirmed.
- **Disk:** ~241 GB free — **not** a constraint.
- **Download started:** `scripts/phase13/fetch_vrsbench_val.py` (huggingface_hub, resumable) began fetching `Images_val.zip`.
- **Measured throughput (direct ranged GET to the resolved CDN `us.aws.cdn.hf.co`):** **1,048,576 bytes in 47.2 s ≈ 0.02 MB/s**.
- **Extrapolated time for 3,792 MB:** **≈ 50 hours** at the observed rate. The `hf_hub` download confirmed this: after ~90 s the on-disk blob was still 0 MB.
- **Action taken:** the stalled download was stopped and the partial `.cache` was removed. `D:\dataset\vrsbench\` is back to just `referring.json`.

## Exact missing asset

| Item | Value |
|---|---|
| Missing file | `Images_val.zip` (VRSBench validation images `P####_####.png`) |
| Size | 3,792 MB (~3.7–3.98 GB) |
| Source | `xiang709/VRSBench` (HuggingFace dataset, public) |
| Reachable? | Yes (HTTP 200) |
| Blocker | **Network throughput ~0.02 MB/s → ~50 h download; not completable here** |
| Disk | 241 GB free (not a blocker) |
| Permission | Provisioning was authorized (not a blocker) |

## Why the evaluation still cannot be completed

Grounding needs the pixels. OWL-ViT must run per (image, referring-expression) and the predicted box compared (IoU@0.5) against the ground truth. Without the images there is nothing to run inference on, and the images cannot be transferred at the available rate. Reporting any IoU would be fabrication.

## Exact procedure to complete once images are provisioned

On a machine/network where the ~3.8 GB downloads at a normal rate (or by copying `Images_val.zip` in out-of-band):

1. **Provision** into `D:\dataset\vrsbench\Images_val\` and verify coverage:
   ```powershell
   $ann = Get-Content D:\dataset\vrsbench\referring.json -Raw | ConvertFrom-Json
   $ids = $ann | Select-Object -ExpandProperty image_id | Sort-Object -Unique
   ($ids | Where-Object { -not (Test-Path "D:\dataset\vrsbench\Images_val\$_") }).Count  # expect 0
   ```
   (Provisioning helper already written: `scripts/phase13/fetch_vrsbench_val.py`.)
2. **Run** the OWL-ViT box-IoU evaluation over `referring.json`:
   - Parse ground-truth box from `ground_truth` tokens / `obj_corner` (normalized), take the top-scoring OWL-ViT box per expression.
   - Use the registered default `score_threshold=0.08` — **do not tune on the test set**.
   - Report: dataset/split (VRSBench referring, val), #evaluated, image/prediction coverage, IoU + IoU@0.5 (label the metric as project-defined if it differs from an official definition), model/checkpoint (`google/owlvit-base-patch32`), processor, device, runtime, failures/skipped.
3. **Save** `reports/phase13/PHASE13_VRSBENCH_GROUNDING_RESULTS.{md,json}`.

## Leakage / integrity (pre-committed)

- Validation images would be used for **inference only** — never training.
- **No** OWL-ViT fine-tuning on VRSBench annotations.
- **No** threshold selection on the test set (registered default used).
- **No** hidden ISRO/SAC data.

## Bottom line

`VRSBench grounding = BLOCKED`. Do **not** convert to PASS. The single remaining blocker is transferring `Images_val.zip` (3,792 MB) over a link that currently runs at ~0.02 MB/s. Everything else (annotations, specialist, disk, procedure, script) is ready.
