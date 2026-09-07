# CDVQA Training Report (Phase 9)

Real training and held-out evaluation on the CDVQA dataset. No synthetic substitution; no
fabricated numbers. Reproduce with the commands below.

## Dataset
- **Source:** https://github.com/YZHJessica/CDVQA (user-provided archive `D:\dataset\CDVQA-main.zip`).
- **SIH status:** SIH26167-prescribed multitemporal change-VQA benchmark.
- **Archive:** 5,162,330 bytes; SHA256 `b4043476ece6ad10b19aa02deb71bfe8094eabbed8a359b9fede500e840ff8db`.
- **Extracted:** `datasets/cdvqa/CDVQA-main/` (QA JSON only).
- **Imagery:** CDVQA ships no images; `file_name`s (0.1524 m) are SECOND dataset images. All
  referenced files are present in the SECOND train archive (verified).
- **Splits (real counts):** Train 65,967 QA / 1,600 images · Val 16,441 / 400 · Test 39,686 / 968
  · Test2 31,036 / 968.
- **Leakage:** Train/Val/Test image sets are disjoint (Train∩Val = Train∩Test = Val∩Test = 0,
  verified). Test2 reuses Test images with harder questions (documented).
- **Question types:** change_or_not, change_ratio_types, increase_or_not, decrease_or_not,
  change_to_what, smallest_change, largest_change, change_ratio.
- **Answer space:** closed vocabulary (19 classes from train): yes/no, land-cover class names,
  change-ratio buckets → a **classification** task (not generative).

## Model
- Frozen CLIP image encoder `flax-community/clip-rsicd-v2` (RS-tuned, cached) → 512-d features
  for T1, T2 (+ T2−T1 difference); trained GRU question encoder + fusion MLP classifier.
- Parameters: 832,275 (all trainable head; encoder frozen). Device: RTX 2050 (CUDA 12.4, torch 2.6).
- Training: 20 epochs, batch 256, Adam lr 1e-3, seed 42; **128 s** (features cached once in ~345 s).

## Results (held-out, evaluated once)
| Split | Accuracy | Majority-per-type baseline |
|---|---|---|
| Val (selection) | 0.7166 | — |
| **Test** | **0.7114** | 0.5084 |
| **Test2** | **0.6632** | 0.4560 |

Per-type (Test): change_or_not 0.838 · decrease_or_not 0.792 · increase_or_not 0.781 ·
change_ratio_types 0.718 · change_to_what 0.583 · largest_change 0.585 · change_ratio 0.394 ·
smallest_change 0.325.

The trained model beats the majority-per-type baseline by ~20 points on both Test and Test2.

## Confidence calibration (real held-out)
Temperature scaling fit on **Val**, evaluated frozen on **Test**: T = 1.1445.
Test ECE **0.0359 → 0.0136**, Brier 0.1612 → 0.1601. Reliability bins track accuracy closely
(e.g. 0.85→0.83, 0.97→0.95). Never fit on test labels. See `reports/benchmarks/cdvqa/calibration.json`.

## Integration
Registered specialist `cdvqa_trained` (registry `MODEL_REGISTRY`), status `degraded`. Checkpoint
reloads and inference verified. It is **not** the default change-VQA route (the deterministic
detector is kept for general imagery — see `docs/phase9_model_selection.md`) because the model is
domain-specific to SECOND 0.1524 m aerial imagery with a closed 19-answer vocabulary.

## Limitations
Domain-specific (SECOND aerial); closed vocab; weakest on counting/ratio (smallest_change 0.33);
frozen-encoder head (not end-to-end); softmax confidence is calibrated but the model is not
validated on other sensors/resolutions.

## Reproduce
```
python scripts/make_dataset_manifests.py
python scripts/train_cd_vqa.py --epochs 20 --batch-size 256 --seed 42
python scripts/calibrate_cdvqa.py
```
Artifacts: `models/phase9/cdvqa/` (checkpoint, vocabs, features, config) ·
`reports/benchmarks/cdvqa/` (metrics, predictions, calibration).
