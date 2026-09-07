# Limitations

An honest, capability-by-capability account of what this system does **not** do or has not
proven. This matters most for the hidden ISRO/SAC evaluation, which will differ from anything
tested here. Nothing below is hidden in prose elsewhere; this page collects it.

Status vocabulary: **IMPLEMENTED** (real, wired, tested) · **EVALUATED** (measured on known
ground truth) · **BENCHMARKED** (measured on a standard external dataset) · **NOT BENCHMARKED**.

## Cross-cutting

- **Nothing is benchmarked on a standard dataset.** Phase 5 adaptation was evaluated on a
  held-out retrieval split; everything else is either evaluated on synthetic ground truth or
  not yet measured. No RSVQA / LEVIR-CC / CDVQA numbers exist here. That is Phase 9.
- **Scores are uncalibrated.** A confidence of "Medium" or a 0.68 score is a derived,
  signal-based judgement, not a calibrated probability of correctness.
- **Every model/tool is registered `degraded`, not `available`.** Real and usable, but not
  accuracy-established.

## Single-image VQA / captioning (Phase 4–5)

- Zero-shot concept scoring against a fixed English vocabulary. A concept outside it cannot be
  reported. No counting, no spatial reasoning, no fine-grained attributes.
- Captions are retrieval-composed from concept scores, not generated prose.
- The Phase 5 LoRA adaptation improved image–text retrieval on a **Lithuania/Summer Sentinel-2**
  subset; benefit on other regions, seasons, and sensors is unestablished. 84% of training
  captions were truncated at CLIP's 77-token limit (see `docs/datasets.md`).

## Grounding (Phase 4)

- OWL-ViT is trained on natural photographs; it transfers to discrete overhead objects and is
  weak on diffuse land cover. A null result is reported as inconclusive, never as absence.
- The coarse-region fallback is tile-level, explicitly not an object outline. No masks.

## Bi-temporal change (Phase 6)

- **Detects change; does not classify change type.** It measures *that* and *where* the imagery
  changed, not *what kind* of change. Cause is never asserted — there is no built-up/urban or
  land-use classifier, and built-up questions state this.
- **NOT BENCHMARKED.** Correctness is verified on synthetic known-truth pairs
  (`test_change_detection.py`); real-world accuracy against a labelled dataset (LEVIR-CD, OSCD)
  is Phase 9.
- **NDVI/NDWI require declared bands.** If the raster does not declare which band is which, no
  index is computed and no vegetation/water trend is claimed — guessing band order would
  fabricate a number.
- **Same modality only.** Optical-vs-SAR differencing is refused; that is Phase 7 fusion.
- **Registration-sensitive.** A pair misregistered by more than one common-grid pixel returns
  "Unable to determine". Sub-pixel misregistration can still add edge noise; coherence filtering
  mitigates but does not eliminate it.
- **Relative radiometric normalisation, not absolute.** No atmospheric correction. A change that
  covers most of a scene can be partly absorbed by normalisation (biased toward
  under-detection); the suspicious-global-change check flags the opposite failure.
- **Two dates only.** No multi-temporal (>2) analysis. The optional land-cover reading of a
  changed region is zero-shot CLIP, labelled indicative, and unavailable if weights are absent.

## Optical–SAR cross-modal fusion (Phase 7)

- **IMPLEMENTED, EVALUATED on synthetic ground truth, NOT BENCHMARKED.** Real feature-level
  fusion (correlation/complementarity, information partition, water agreement, SAR-only
  structure regions); accuracy on a public benchmark is Phase 9.
- **SAR is uncalibrated.** Reported as log-scaled backscatter unless the source metadata declares
  calibration; sigma0/gamma0/dB is never assumed. SAR evidence is measured backscatter/roughness,
  never a semantic land-cover class.
- **Exactly one optical + one SAR image** required; two-optical, two-SAR, or missing-modality
  pairs are refused. Cross-modality differencing is never faked.
- **NDVI/NDWI and water agreement need declared optical bands**; otherwise declined with a reason.
- **Correlation is withheld** when a modality is near-uniform (would be spurious).
- **No learned fusion model.** The only learned component is an optional zero-shot CLIP
  land-cover reading of the optical image (indicative, never applied to SAR).
- **No sub-pixel co-registration refinement**; relies on the georeferencing/overlap of the inputs.

## Data and generalisation

- Test fixtures are synthetic rasters; they verify behaviour, not accuracy.
- Training data provenance and caveats (including an unofficial imagery mirror) are in
  `docs/datasets.md`. Annotations are not human-verified.
- The hidden ISRO/SAC set (Cartosat-2S / RISAT) is unavailable to us and, by rule, is never used
  for training. No hard-coded answers, coordinates, or demo-only routing exist.

## Operational

- Network-exposed API endpoints are **unauthenticated** (documented in the route module). Add
  authentication before any shared or production deployment.
- The Docker image builds have not been run on this machine (low disk); `docker compose config`
  is validated. Change detection needs no GPU; the image bundles CPU torch + peft for the Phase
  5 adapter.
