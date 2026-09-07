# Phase 7 — Optical + SAR cross-modal analysis

What was built, why, and what it does and does not claim. Every number is reproducible from the
code. Nothing here is benchmark-validated — no cross-modal benchmark has been run (see
*Evaluation*).

Status vocabulary: **IMPLEMENTED** (real, wired, tested) · **EVALUATED** (measured on known
synthetic ground truth) · **BENCHMARKED** (measured on a standard external dataset — **none in
Phase 7**) · **CALIBRATED** (confidence mapped to correctness — no) · **NOT YET VALIDATED**.

---

## What ships

| Capability | Status |
|---|---|
| Optical+SAR pair validation (exactly one of each modality) | IMPLEMENTED, EVALUATED |
| Co-registration to a common grid (reuses Phase 6 `align_modalities`) | IMPLEMENTED, EVALUATED |
| SAR preprocessing (log-scaled backscatter, uncalibrated by default) | IMPLEMENTED |
| Optical feature extraction (brightness, spectral variance, texture, NDVI/NDWI) | IMPLEMENTED |
| **Real feature-level cross-modal fusion** (correlation/complementarity, information partition) | IMPLEMENTED, EVALUATED |
| Cross-modal water agreement (optical NDWI vs SAR dark surfaces) | IMPLEMENTED, EVALUATED |
| Complementary regions (SAR-only structure, water agreement) with coordinates | IMPLEMENTED, EVALUATED |
| Cross-modal description + cross-modal VQA | IMPLEMENTED |
| Fusion visualizations (composite, complementary map, aligned optical/SAR) | IMPLEMENTED |
| Agent routing, registry, evidence-based confidence, trace | IMPLEMENTED |
| Frontend cross-modal view + evidence renderers | IMPLEMENTED |
| Accuracy on a cross-modal benchmark | **NOT BENCHMARKED** — Phase 9 |

## Why this is genuine fusion, not caption concatenation

The SIH requirement forbids "optical→caption + SAR→caption, concatenated". This implementation
never does that. It builds a **joint feature representation** on one grid and measures quantities
that *neither modality produces alone*:

- **Cross-modal correlation / complementarity.** Pearson correlation between optical brightness
  and SAR backscatter over the shared valid pixels; complementarity = 1 − |correlation|. High
  complementarity means the two modalities carry genuinely different information here. This
  number cannot exist for a single modality.
- **Information partition.** Of the structured pixels, how much is structured in *optical only*,
  *SAR only*, or *both* — a direct measure of what each modality uniquely contributes.
- **Cross-modal water agreement.** Open water is optically distinct (NDWI-positive) and SAR-dark
  (specular reflection away from the sensor). The intersection-over-union of optical-water and
  SAR-dark pixels is a two-physics corroboration; the disagreement (SAR-dark but not
  optical-water) is complementary information (roads, shadows, radar-smooth surfaces).
- **SAR-only structure regions.** Coherent areas rough in SAR but spectrally flat in optical —
  literally "what SAR reveals that optical does not", as connected components with coordinates.

That is feature/evidence-level fusion. The chosen method and its justification are recorded in
the `ml/inference/fusion.py` module docstring and the registry entry.

## Why algorithmic, not a large multimodal model

The 4 GB RTX 2050 rules out a 2–3 B multimodal VLM, but that is not the main reason. There is no
open, licensed, drop-in optical+SAR fusion model that (a) fits, (b) is reproducible, and (c)
would not fabricate confident semantics on arbitrary sensors/resolutions — the failure this
project forbids. A deterministic feature-level analysis instead measures real physical
quantities at any resolution and reports its own uncertainty. It needs no weights and runs in
CI. A learned fusion model can be added later behind the same specialist interface.

The **only** learned component is an optional zero-shot CLIP land-cover reading of the *optical*
image, labelled indicative and never applied to SAR (CLIP was not trained on radar).

## SAR preprocessing and honesty about units

SAR band(s) are averaged in the linear domain and log10-scaled — the standard representation,
because radar power is multiplicative and speckle behaves additively in the log domain. The
result is **robust-normalised, uncalibrated backscatter** unless the source metadata declares
calibration (tokens like `sigma0`, `gamma0`, `calibrat`). sigma0/gamma0/dB is **never assumed**.
SAR evidence is always stated as measured backscatter and roughness — never as land-cover
classes ("buildings", "water") on SAR alone.

## Optical preprocessing

Reuses the Phase 2/6 raster stack. Brightness and per-pixel spectral variance are always
available; NDVI (red+NIR) and NDWI (green+NIR) are computed **only when the raster declares its
band roles** (`geospatial.resolve_band_roles`). Band order is not standardised, so guessing it
would produce a plausible false index — the code refuses.

## Co-registration

Phase 6's grid planner was extracted so both `align_pair` (unchanged) and the new
`align_modalities` share it. `align_modalities` keeps **all** optical bands and **all** SAR bands
(fusion needs the optical spectral bands and the SAR backscatter together, so the Phase 6
min-band truncation would be wrong). It reprojects a CRS mismatch, resamples to the **coarser**
resolution so neither modality is upsampled, crops to the overlap, and caps the grid at 768 px.
Every operation is recorded and surfaced as a `Co-registration` trace step. A pair that cannot be
aligned raises and is refused.

## Honesty mechanisms (anti-fabrication, made concrete)

- Exactly one optical + one SAR image required; two-optical, two-SAR, or missing-modality pairs
  are refused with an explanation, not fused anyway.
- SAR is uncalibrated backscatter unless metadata says otherwise; no sigma0/gamma0 claim.
- No semantic SAR classes; "built structures in SAR?" reports measured roughness and states no
  built-up classifier exists.
- NDVI/NDWI and water agreement need declared bands, else declined with the reason.
- Correlation/complementarity withheld when a modality is near-uniform (would be spurious).
- Coordinates only when georeferencing is reliable; area in m² only for metric CRSs.
- Confidence is evidence-derived and uncalibrated; unanswerable questions → "Unable to
  determine".
- Every rendered pixel corresponds to a measured condition; no decorative fusion maps.

## Evaluation

**Measured (EVALUATED).** `backend/tests/test_cross_modal.py` asserts against a synthetic pair
with known features: `align_modalities` keeps 3 optical + 1 SAR band; the water strip is agreed
on across modalities (IoU > 0.2); the structured block appears as a SAR-only region with
coordinates; correlation is withheld for a uniform modality; NDVI/water are refused without
declared bands; SAR is reported uncalibrated. These prove correctness, not real-world accuracy —
the imagery is synthetic.

**NOT BENCHMARKED.** No public optical+SAR fusion benchmark has been downloaded or run. Suitable
public benchmarks for *cross-modal complementarity* (as opposed to SAR-only or fusion-for-
classification) are scarce; Phase 9 will either evaluate on a suitable dataset or document the
absence of one. No fusion accuracy figure is claimed anywhere.

## Performance (measured, CPU)

| Input each modality | Common grid | Align | Analyze | Render | Peak memory |
|---|---|---|---|---|---|
| 512×512 | 512×512 | 42 ms | ~1.25 s first call† | 128 ms | 33 MB |
| 1024×1024 | 768×768 | 154 ms | 236 ms | 174 ms | 53 MB |
| 2048×2048 | 768×768 | 351 ms | 236 ms | 170 ms | 53 MB |

† The first `analyze` pays a one-off `scipy.ndimage` import cost; subsequent calls are ~230 ms.
Peak memory stays ~50 MB regardless of input size because the common grid is capped at 768 px and
reads go through `WarpedVRT`. No GPU is used.

## Example outputs (live API, synthetic imagery)

- **"Analyze these optical and SAR images together."** → *The two modalities are largely
  redundant here (correlation 0.85, complementarity 0.15). Optical brightness averages 0.76 with
  6.2% structured; SAR (log10-scaled backscatter, uncalibrated) averages 0.50 with 10.1%
  structured. SAR shows structure the optical does not over 3.9% of the scene. Optical and SAR
  agree on water over 15.6% of the scene (IoU 0.62). …*
- **"What does SAR reveal that optical does not?"** → reports SAR-only structured fraction,
  dark/bright fractions, named regions with coordinates, and notes SAR is illumination-
  independent.
- **"Which modality provides stronger evidence of vegetation?"** → *Optical provides the more
  direct vegetation evidence (mean NDVI …); SAR measures backscatter/roughness, not greenness.*
- **"Is there evidence of construction?"** → reports measured structure, then *no built-up
  classifier was applied to either image*.
- **Two optical images** → rejected: *needs exactly one optical and one SAR image*.

## Reproduce

```powershell
pip install -e .   # pulls scipy; no GPU or weights needed
.\.venv\Scripts\python.exe -m pytest backend/tests/test_cross_modal.py -q
.\.venv\Scripts\python.exe -m pytest backend/tests/test_fusion_api.py -q
```

## What remains

- **Phase 9:** find/run a suitable cross-modal benchmark or document its absence; calibrate
  confidence; only then can the registry status move off `degraded`.
- **Not scheduled:** a learned fusion model behind the same interface; multi-band SAR
  polarimetric decomposition; true co-registration refinement (sub-pixel) for optical/SAR.
