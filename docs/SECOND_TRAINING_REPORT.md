# SECOND Training Report (Phase 9)

Real training and held-out evaluation of a learned change detector on the SECOND dataset. No
synthetic substitution; no fabricated numbers.

## Dataset classification (important)
**SECOND is an ADDITIONAL PUBLIC dataset used to train / externally validate the
change-detection component. It is NOT SIH-provided, NOT ISRO-provided, and NOT an official SIH
benchmark.**

- **Source:** user-provided archive `D:\dataset\SECOND_train_set.rar` (SECOND — SEmantic Change
  detectiON Dataset, train set).
- **Archive:** 2,406,111,691 bytes (~2.24 GB); SHA256
  `5ee2a82b5824b3f5e3c5bfaf018835862e7623993ff9ac5ce57fd11fcfad2b4e`.
- **Extracted:** `D:\dataset\second\` via `tar` (libarchive RAR read). Original archive untouched.
- **Structure:** `im1/` (T1 RGB), `im2/` (T2 RGB), `label1/`, `label2/` (per-date semantic
  labels, RGB-coded), **2,968 pairs**, 512×512.
- **Classes (6 land-cover + no-change):** nvg_surface (128,128,128), tree (0,128,0),
  low_vegetation (0,255,0), water (0,0,255), buildings (128,0,0), playground (255,0,0);
  no-change = white (255,255,255).

## Task and label derivation (documented, not silent)
The provided archive is the **train set only** (no official val/test). Splits were created from
the 2,968 pairs with a fixed seed (42), 70/15/15 → **train 2,077 · val 445 · test 446**
(disjoint; sample ids saved to `reports/benchmarks/second/splits.json`). The held-out split is an
**INTERNAL** split, NOT the official SECOND test set.

Learning target: **binary change** (changed vs unchanged), derived from the semantic labels as
`changed = (label1 != no-change/white)`. This is an explicit, lossy reduction of the semantic
labels chosen so the learned detector is directly comparable to the deterministic Phase 6 CVA.
Full from-to semantic change is preserved in the raw labels and left as future work.

## Model
- Compact Siamese U-Net over the 6-channel concatenation of T1+T2 (256×256), Dice+BCE loss, AMP.
- Parameters: 1,929,281. Device: RTX 2050 (CUDA 12.4, torch 2.6). 12 epochs, batch 8, seed 42.
- Training time: **~1,639 s** (~27 min). Best checkpoint selected on val IoU.

## Results (held-out internal test, evaluated once)
| Detector | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| **Learned Siamese U-Net** | **0.4219** | 0.5736 | 0.6147 | **0.5934** |
| Deterministic reference (Otsu on |T2−T1|, in the spirit of Phase 6 CVA) | 0.2248 | 0.3076 | 0.4549 | 0.3670 |

The learned detector **nearly doubles** the deterministic reference IoU (0.42 vs 0.22) and lifts
F1 from 0.37 to 0.59 on the same held-out pairs.

## Baseline-comparison honesty
The reference is a deterministic radiometric-difference baseline (Otsu threshold on the L2
difference magnitude), computed on the same held-out pairs — a fair stand-in for the Phase 6 CVA
family on non-georeferenced RGB PNGs. It is labelled a reference, not the exact production
`cva-robust` configuration (which targets georeferenced rasters). Neither baseline nor learned
model used test labels for tuning.

## Integration
Registered specialist `second_change_learned` (status `degraded`); checkpoint reloads and
inference verified on a real pair. **Not** the default temporal-change route — the deterministic
CVA remains default for general/georeferenced imagery (see `docs/phase9_model_selection.md`).

## Limitations
Binary change only (not from-to semantic); trained on 0.1524 m aerial RGB (generalization to
other sensors/resolutions unestablished); internal (not official) held-out split; short training
budget (12 epochs) — the number is real but not a tuned state-of-the-art.

## Reproduce
```
tar -xf D:\dataset\SECOND_train_set.rar -C D:\dataset\second
python scripts/make_dataset_manifests.py
python scripts/train_second_change.py --epochs 12 --batch-size 8 --size 256 --seed 42
```
Artifacts: `models/phase9/second/` (checkpoint, config) ·
`reports/benchmarks/second/` (metrics, splits, config).
