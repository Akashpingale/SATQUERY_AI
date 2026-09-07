# Phase 12 — VRSBench Grounding: BLOCKED

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-06
**Status:** `BLOCKED` (unchanged from Phase 11 — re-verified, not fabricated)

## Verdict

VRSBench referring-expression **grounding cannot be quantitatively evaluated** in this environment because the **validation image archive is not provisioned**. The grounding specialist and the annotations are ready; only the images are missing. No IoU result is reported, because none was measured.

## What IS available (verified this phase)

- **Annotations present:** `D:\dataset\vrsbench\referring.json` (9.8 MB).
  - **16,159** referring expressions over **9,318** distinct images.
  - Per-record fields (verified): `image_id` (e.g. `P0003_0002.png`), `question` (referring expression), `ground_truth` (quantized box tokens, e.g. `{<25><40><33><60>}`), `obj_corner` (8 normalized 0–1 corner floats), `obj_cls`, `type`, `size_group`.
- **Grounding specialist ready:** `grounding` = `google/owlvit-base-patch32` (Apache-2.0), open-vocabulary box localization, wired into the live workflow and exercised in the demo (Flow-level grounding overlays render). OWL-ViT emits boxes → **box IoU@0.5 is architecturally compatible** with the referring-bbox annotations.
- **Disk space is NOT the blocker:** drive D: has **~241 GB free**; the ~3.98 GB archive would fit.

## Exact missing asset

| Item | Value |
|---|---|
| Missing file | VRSBench `Images_val.zip` (validation images `P####_####.png`) |
| Referenced by | `image_id` in `referring.json` (9,318 distinct filenames) |
| Expected size | **~3.98 GB** (established in Phase 11) |
| Present locally? | **No** — `D:\dataset\vrsbench\` contains only `referring.json` |
| Reason not fetched | Large download not auto-permitted; must be explicitly provisioned |

## Why the evaluation cannot currently be completed

The referring annotations give expression + ground-truth box + `image_id`, but grounding requires the **pixels**: OWL-ViT must run on each `P####_####.png` to predict boxes that are then compared (IoU@0.5) against the ground-truth. With the images absent, there is nothing to run inference on, and reporting any IoU would be fabrication.

## Exact procedure to run once images are provisioned

1. **Provision images** into `D:\dataset\vrsbench\Images_val\` (unzip `Images_val.zip`) so that every `image_id` in `referring.json` resolves to a file. Verify:
   ```powershell
   $ann = Get-Content D:\dataset\vrsbench\referring.json -Raw | ConvertFrom-Json
   $ids = $ann | Select-Object -ExpandProperty image_id | Sort-Object -Unique
   $missing = $ids | Where-Object { -not (Test-Path "D:\dataset\vrsbench\Images_val\$_") }
   "images referenced: $($ids.Count); missing: $($missing.Count)"   # expect missing = 0
   ```
2. **Integrity check** (as far as possible): confirm files are valid PNGs and non-empty; spot-check a few dimensions against `size_group`/`obj_corner` scale.
3. **Run OWL-ViT box-IoU evaluation** over `referring.json`:
   - Parse the ground-truth box from `ground_truth` tokens (or `obj_corner` → axis-aligned bbox), normalized to the image size.
   - For each expression, run the `grounding` specialist (OWL-ViT) with the expression as the query; take the top-scoring box.
   - Compute IoU; count IoU@0.5 hits; record successes/failures.
   - **Do NOT tune the score threshold on this held-out set** (use the registered default `score_threshold=0.08`).
4. **Record** (per Phase 12 reproducibility): dataset/split (VRSBench referring, val), number evaluated, model/checkpoint (`google/owlvit-base-patch32`), preprocessing (normalization, box parsing), IoU metric, IoU@0.5, successful/failed sample counts, and limitations. Save to `reports/phase12/PHASE12_VRSBENCH_GROUNDING_REPORT.md`.

Suggested command (script to be added when images exist):
```powershell
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; $env:HF_HOME="D:\SatQuery-AI\models\hf-cache"
.\.venv\Scripts\python.exe scripts\phase12\eval_vrsbench_grounding.py `
  --annotations D:\dataset\vrsbench\referring.json `
  --images D:\dataset\vrsbench\Images_val `
  --iou-threshold 0.5 --score-threshold 0.08 --limit 0
```

## Bottom line

- **Do not** convert this to `PASS`. Grounding is functionally wired but its public-benchmark IoU is **unmeasured**.
- The **only** missing input is the ~3.98 GB image archive. Once provisioned, the procedure above produces a reproducible IoU result with no code changes to the specialist and no threshold tuning on the test set.
