# Phase 6 — Bi-temporal change understanding

What was built, why it was built this way, and what it does and does not claim. Every number
in this document is reproducible from the code; nothing here is a benchmark result, because no
change-detection benchmark has been run (see *Evaluation* below and `docs/evaluation.md`).

Status vocabulary used throughout, kept strict:

- **IMPLEMENTED** — real code, wired into the API and UI, covered by tests.
- **EVALUATED** — measured against known ground truth (here, synthetic pairs with a known
  changed area).
- **BENCHMARKED** — measured against a standard external dataset. **Nothing in Phase 6 is
  benchmarked.**
- **NOT YET VALIDATED** — implemented but not measured for accuracy.

---

## What ships

| Capability | Status |
|---|---|
| Two-image temporal input (T1/T2), validation, safe co-registration | IMPLEMENTED, EVALUATED on synthetic pairs |
| Real change detection (change vector analysis / SAR log-ratio) | IMPLEMENTED, EVALUATED on synthetic pairs |
| Change description (grounded, measurement-based) | IMPLEMENTED |
| Change-based VQA | IMPLEMENTED |
| Change map + T1/T2/change strip evidence | IMPLEMENTED |
| NDVI / NDWI change, when bands are declared | IMPLEMENTED, EVALUATED |
| Agent routing + registry + trace + evidence-based confidence | IMPLEMENTED |
| Frontend temporal workflow (compare + change views) | IMPLEMENTED |
| CDVQA-style evaluation interface | IMPLEMENTED (interface only; **not** run against CDVQA) |
| Accuracy on a standard change benchmark | **NOT BENCHMARKED** — Phase 9 |

Both mandatory SIH capabilities were implemented, not just one: change **description** and
change **VQA**, plus the optional change **map**.

## The approach, and why

### Chosen: radiometrically normalised change vector analysis (CVA)

Change is measured, not classified:

```
align to a common grid (geospatial.align_pair)
  → estimate misregistration          (are we comparing the same pixels?)
  → robust radiometric normalisation   (remove illumination/atmosphere differences)
  → change magnitude                    (CVA for optical, |log-ratio| for SAR)
  → robust scaling                      (comparable across scenes)
  → threshold = robust noise floor      (median + 3·MAD-sigma)
  → binary opening                      (drop speckle)
  → connected components ≥ min size     (keep only spatially coherent change)
  → per-region geometry, area, brightness, and NDVI/NDWI where bands are declared
```

Every stage is deterministic: the same two rasters and the same config always produce the same
mask. Full method detail is in the module docstring of `ml/inference/temporal.py`.

### Why not a pretrained deep change-detection model

The obvious alternative is a LEVIR-CD-trained network (BIT, ChangeFormer, TinyCD). It was
rejected for concrete, checkable reasons:

1. **Resolution and domain mismatch.** Those checkpoints are trained on ~0.5 m aerial RGB and
   almost exclusively on **building** change. Applied to arbitrary satellite resolutions and
   land-cover types they emit confident, plausible, wrong masks — the exact failure this
   project forbids. The hidden ISRO/SAC set will not match their training distribution.
2. **Distribution.** They ship as loose per-repository checkpoint files (Google Drive, Baidu,
   Dropbox), not through a package index, so pinning a verifiable, licensed artifact is not
   clean.
3. **The 4 GB constraint.** Not the binding reason — several would fit — but it removed any
   upside from a heavier model.

CVA instead measures a real physical quantity at any resolution and **reports its own
uncertainty** (registration quality, coherence, changed fraction), which is what lets the
system say "unable to determine" honestly. A learned model can still be added later behind the
same specialist interface without touching the API.

### Why the detector is algorithmic, and what that buys

No weights, no GPU, no network. Change detection runs in CI unconditionally and on any machine.
The only learned component is an **optional** land-cover reading of the largest changed region
(zero-shot CLIP, the Phase 4/5 model), used strictly to describe what a region looks like —
never to decide whether it changed, and always labelled indicative.

## Honesty mechanisms (the anti-fabrication requirement, made concrete)

- **A pixel is only "changed"** if its measured magnitude clears the threshold *and* it belongs
  to a region that survived coherence filtering. Rendered tint is drawn only over such pixels;
  a test asserts no tint appears outside a measured region.
- **Misregistration is refused, not described.** If the pair correlates best at a shift of more
  than one pixel and that shift explains most of the decorrelation, the result is
  `inconclusive` with an explanation. Displacement artifacts are never reported as change.
- **Optical vs SAR is refused.** Differencing two different physical quantities would be
  meaningless, so the combination returns a stated limitation.
- **NDVI/NDWI only from declared bands.** Band order is not standardised; guessing it would
  produce a plausible false index. Undeclared rasters get "not available" with the reason.
- **Cause is never asserted.** The strongest statement is a labelled before/after land-cover
  reading. "Construction", "new buildings", and "roads" are never claimed — there is no
  built-up classifier, and built-up questions say so explicitly.
- **Area in m² only for metric CRSs.** A geographic (degree) CRS yields a percentage and a
  pixel count, never a fabricated square-metre figure.
- **Confidence is derived from measured signals** (registration, coherence, changed-fraction
  margin, magnitude headroom); an unanswerable question yields "Unable to determine", not a
  number.
- **Resampling is recorded in the trace**, satisfying the geospatial rule that transformations
  applied to a user's data must be visible.

## Resource usage (measured)

| | |
|---|---|
| Device | CPU only for detection; the optional land-cover reading uses the existing CLIP model |
| Dependencies added | `scipy>=1.11` (connected components, morphology). scikit-image/OpenCV rejected as heavier for the same three functions |
| Memory | Aligned arrays capped at 768 px longest side via `WarpedVRT` decimation; a large pair is downsampled, never loaded whole |
| Determinism | Fully deterministic; no seed needed |
| GPU | Not required, not used, by design |

## Configuration and reproducibility

All thresholds live in `ml.inference.temporal.ChangeConfig` as named constants, none hidden:
`noise_sigma=3.0`, `opening_size=3`, `min_region_pixels=25`, `max_tolerable_shift_pixels=1`,
`min_changed_fraction=0.005`, `suspicious_changed_fraction=0.60`, `min_valid_fraction=0.20`.
Three are exposed as allowlisted agent parameters (`noise_sigma`, `min_region_pixels`,
`min_changed_fraction`) so a plan can tune them within validated ranges. The algorithm version
`cva-robust-1.1.0` is recorded in the registry, the trace, and every evidence payload, so a
result is always tied to a method version.

## Evaluation

**Measured (EVALUATED).** `backend/tests/test_change_detection.py` asserts measured behaviour
against known ground truth on synthetic pairs: a 30×40 changed block is measured as exactly
1200 changed pixels; an identical pair measures zero; an illumination-only difference measures
zero; a dominant change does not hide a weaker one; a 2 px shift is reported inconclusive; NDVI
falls when NIR collapses. These verify correctness, not real-world accuracy.

**Not benchmarked.** No standard change-detection or CDVQA dataset has been downloaded or run.
`ml/evaluation/change_vqa.py` provides a benchmark-ready interface — a documented sample schema,
a predictor that drives the real pipeline, and metrics that report **coverage alongside
accuracy** so a system that declines most questions cannot look strong. The concrete CDVQA field
names and question taxonomy are **unverified** (the dataset was not obtained), so converting a
real release into the documented JSONL is an explicit adapter step. Running it is Phase 9 work.
Any CDVQA accuracy figure stated before that would be fabricated.

## Example outputs (from the live API, synthetic imagery)

Real answers produced by the pipeline. The imagery is synthetic, so the *content* is not
meaningful; what is real is that each figure is measured from the pixels.

> **"What changed between these two images?"** — *Change affects 10.7% of the shared area
> (about 275,000 m²), concentrated in the northern western quadrant (centred near 13.5644,
> 75.0048), and the changed areas are brighter in the later image. … the cause is not
> established.*

> **"Has vegetation decreased between these dates?"** (bands declared) — *NDVI decreased from
> 0.622 to 0.489 (change -0.133) … Within the changed areas it is lower by 0.872 … That supports
> a decrease in vegetation.*

> **"Has vegetation decreased?"** (bands undeclared) — *cannot be determined from this pair: red
> and near-infrared bands are not both declared … inferring band order would produce a plausible
> but unfounded number.* → confidence **Unable to determine**.

> **"Is there evidence of construction?"** — reports the measured change, then: *no built-up
> classifier was trained or evaluated, so the system reports the measured change and its
> land-cover reading rather than a land-use verdict.*

> **Misregistered pair** — *Unable to determine change from the available evidence. The two
> images are misregistered by about 2 pixels …* → confidence **Unable to determine**.

## Reproduce

```powershell
pip install -e .   # pulls scipy; no GPU or weights needed for change detection

# Unit tests: measured behaviour against known synthetic ground truth
.\.venv\Scripts\python.exe -m pytest backend/tests/test_change_detection.py -q

# Integration: registry, planner, specialists, API, evidence, trace, negative cases
.\.venv\Scripts\python.exe -m pytest backend/tests/test_temporal_api.py -q
```

## What remains

- **Phase 7:** genuine optical-SAR fusion. Phase 6 refuses cross-modality differencing; fusion
  is a separate capability.
- **Phase 9:** run the CDVQA interface against the real dataset; benchmark change-detection
  quality; calibrate confidence. Only then can the registry status move off `degraded`.
- **Not scheduled:** a learned change-detection model behind the same interface;
  multi-temporal (>2 date) analysis; semantic change-type classification.
