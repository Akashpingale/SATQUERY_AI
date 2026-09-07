# Models — provenance, licensing, and limitations

Every model actually used by SatQuery AI is listed here with its real provenance. Nothing
in this file is aspirational: if a capability is not backed by a working model, it is
recorded as such.

## Current models

| Capability | Model | Params | License | Status |
|---|---|---|---|---|
| VQA, Captioning (baseline) | [`flax-community/clip-rsicd-v2`](https://huggingface.co/flax-community/clip-rsicd-v2) | 151.3 M | **Not declared upstream** — see model card | `degraded` |
| **VQA, Captioning (adapted)** | **`clip-rsicd-v2` + Phase 5 LoRA adapter** | 1.97 M trainable of 153.2 M | Inherits base model + CDLA-Permissive-1.0 (data) | `degraded` |
| Grounding | [`google/owlvit-base-patch32`](https://huggingface.co/google/owlvit-base-patch32) | 153.2 M | Apache-2.0 | `degraded` |
| **Temporal change / change VQA** | **Algorithmic — no model** (change vector analysis) | — | OSS (NumPy/SciPy/Rasterio) | `degraded` |
| **Optical+SAR cross-modal** | **Algorithmic — no model** (feature-level fusion) | — | OSS (NumPy/SciPy/Rasterio) | `degraded` |
| **Change VQA (prototype default)** | **`change_vqa_blip` — external BlipForQuestionAnswering, CDVQA/SECOND fine-tune** (`change_model/`) | ~385 M | External checkpoint (out-of-band) | `degraded` |
| **Captioning (prototype default)** | **`captioning_blip` — external BlipForConditionalGeneration, VRSBench fine-tune** (`caption_model/`) | ~247 M | External checkpoint (out-of-band) | `degraded` |

The two Hugging Face models are ungated: no token is required. Weights download on first use
into `HF_HOME` (~1.2 GB combined) and are never committed. The adapter is produced locally
(12.3 MB) and is also not committed — it is reproducible from `scripts/train_phase5.py`.

## Phase 6 change detection — algorithmic, not a model

The `temporal_change` and `change_vqa` tools are registered like any other specialist, but they
are **deterministic algorithms, not learned models**: `checkpoint` is `None` because there is
nothing to load, not because something is missing. Method: radiometrically normalised change
vector analysis (SAR log-ratio), robust noise-floor thresholding, and spatial-coherence
filtering — full detail in `docs/phase6.md` and the `ml/inference/temporal.py` docstring. They
run on CPU with no weights and no network.

Status is `degraded`, not `available`, for one honest reason: the method is real and
reproducible but has **not been benchmarked** against a labelled change-detection or CDVQA
dataset (Phase 9). Version `cva-robust-1.1.0` is recorded in the registry, the trace, and every
evidence payload.

The **only** learned component in the temporal path is an optional zero-shot land-cover reading
of the largest changed region, which reuses the CLIP model above. It describes what a region
looks like and is labelled indicative; it never decides whether change occurred.

> **Why not a pretrained change-detection network (BIT/ChangeFormer/TinyCD):** those checkpoints
> are trained on ~0.5 m aerial building change and produce confident, wrong masks at other
> resolutions and land-cover types — the failure this project forbids. See `docs/phase6.md` for
> the full decision. A learned model can be added later behind the same specialist interface.

## Phase 7 optical+SAR cross-modal — algorithmic, not a model

The `optical_sar_fusion` tool is a **deterministic feature-level analysis**, not a learned
model. It co-registers the optical and SAR images and measures cross-modal quantities neither
produces alone: correlation/complementarity, the optical-only / SAR-only / shared information
partition, optical↔SAR water agreement, and SAR-only structure regions. This is genuine fusion,
not caption concatenation. Method: `ml/inference/fusion.py`; rationale: `docs/phase7.md`.

Status `degraded`: real and reproducible, not benchmarked. SAR is **log-scaled, uncalibrated
backscatter** unless the source metadata declares calibration — sigma0/gamma0 is never assumed —
and SAR evidence is reported as measured backscatter/roughness, never as land-cover classes.
Version `xmodal-1.0.0`. The only learned component is an optional zero-shot CLIP land-cover
reading of the **optical** image (indicative, never applied to SAR).

## Phase 5 adapted model

Real LoRA adaptation of the baseline on **8,362 BigEarthNet.txt image–caption pairs**. This is
genuine adaptation on remote-sensing data, not a generic model with a "remote sensing" prompt.

| | |
|---|---|
| Base | `flax-community/clip-rsicd-v2` (CLIP ViT-B/32) |
| Method | LoRA (r=16, α=32, dropout 0.05) on `q_proj,k_proj,v_proj,out_proj` in **both** towers |
| Objective | Symmetric InfoNCE, the loss CLIP itself is trained with |
| Data | BigEarthNet.txt Lithuania/Summer captioning subset — official annotations, `docs/datasets.md` |
| Trainable | 1,966,080 of 153,243,393 (**1.283 %**) |
| Hardware | NVIDIA GeForce RTX 2050, 4 GB — 7.6 min, peak 2,219.2 MB VRAM |
| Artifact | `models/phase5/adapter/` — 12.3 MiB total; `adapter_model.safetensors` 7,892,256 bytes |
| Version string | `clip-rsicd-v2+lora-ben-txt-r16a32` |

Full method and hyperparameters: `docs/training.md`.

### Measured improvement (held-out test split, n=2,053)

Image↔text retrieval, both models evaluated in one run with identical preprocessing. Chance
R@1 for this pool is 0.00049.

| Metric | Baseline | Adapted |
|---|---|---|
| image→text R@1 / R@5 / R@10 | 0.0054 / 0.0166 / 0.0253 | **0.0346 / 0.0867 / 0.1432** |
| text→image R@1 / R@5 / R@10 | 0.0015 / 0.0058 / 0.0146 | **0.0185 / 0.0697 / 0.1218** |
| mean recall | 0.0115 | **0.0791** (+0.0675, **6.9×**) |
| median rank image→text / text→image | 518 / 645 | **85 / 84** |

Every recall metric improved and both directions improved together. Absolute values are still
low — the candidates are 2,053 near-identical Lithuanian summer scenes and 84 % of captions
are truncated by CLIP's 77-token limit. `docs/evaluation.md` explains what this does and does
not establish; **no VQA or captioning accuracy claim is made from it**.

### How it is served

Baseline and adapted are **separate registry entries** (`vqa`/`captioning` and
`vqa_adapted`/`captioning_adapted`), so every plan, trace, and evidence record states
unambiguously which one ran. The adapter is located by `PHASE5_ADAPTER_DIR`. If it is missing
or unloadable, the backend serves the baseline and the trace records the reason for the
fallback — an adapted model is never reported when it did not load.

**Grounding is not adapted.** No grounding-specific training or evaluation was performed, so
no grounding improvement is claimed, and a test asserts that grounding has no adapted
preference.

> **Licensing caveat.** `clip-rsicd-v2` declares no license in its Hugging Face metadata.
> It is a community fine-tune of OpenAI CLIP on the RSICD dataset. Confirm the licensing
> terms of the base model, the fine-tune, and RSICD before any commercial or redistributed
> use. This is recorded rather than assumed.

## Prototype trained BLIP models (change_vqa_blip, captioning_blip)

Two externally-trained BLIP checkpoints supplied by the team are integrated as **additional**
specialists and made the **prototype defaults** for their tasks. All prior routes (deterministic
change VQA, `cdvqa_trained`, `cdvqa_transfer`, and the CLIP-based captioner) remain registered
and selectable; routing falls back to them automatically if a BLIP checkpoint is unavailable, so
nothing breaks.

| | `change_vqa_blip` | `captioning_blip` |
|---|---|---|
| Architecture | `BlipForQuestionAnswering` (single-image, generative) | `BlipForConditionalGeneration` (single-image, generative) |
| Checkpoint | `change_model/model.safetensors` (read-only) | `caption_model/model.safetensors` (read-only) |
| Training data | CDVQA + SECOND (external fine-tune) | VRSBench (external fine-tune) |
| Task | `change_vqa` | `captioning` |
| Default route? | Yes, when `CHANGE_VQA_MODEL=blip` (prototype default); falls back to deterministic | Yes, when `CAPTIONING_MODEL=blip` (prototype default); falls back to CLIP captioner |
| Benchmark | CDVQA held-out **Test 0.5467 / Test2 0.5033** (below `cdvqa_trained` 0.7114/0.6632) | **Not benchmarked** — only training-loss curves available |
| Honesty | It is single-image; a T1/T2 pair is shown under the frozen `t1` convention. Not a two-image temporal detector | Generative free text; confidence is an uncalibrated mean token probability |

Selection is config-driven (`CHANGE_VQA_MODEL`, `CAPTIONING_MODEL`); the resolved model_id is
shown in the result card ("Selected tool" + "Our trained model" badge), the plan, the trace, and
the evidence. Weights are loaded read-only and never modified. **No accuracy improvement is
claimed for these over the existing routes** — `change_vqa_blip` scores below `cdvqa_trained` on
CDVQA, and `captioning_blip` has no measured caption metric; they are the prototype's demonstrated
trained contribution, made default by request.

## Why these models

`clip-rsicd-v2` is CLIP fine-tuned on **RSICD**, a remote-sensing image captioning dataset,
so it is genuinely remote-sensing-adapted rather than a purely natural-image model. It runs
comfortably on CPU, which is what serving targets by default. It also fits LoRA training in
4 GB of VRAM, which is what made the Phase 5 adaptation possible on the available GPU, and it
is the model the Phase 4 specialists already read embeddings from — so adapting it improves
the shipping product instead of adding a parallel model nothing uses.

`owlvit-base-patch32` provides open-vocabulary detection, which is what makes real bounding
boxes possible from a free-text target phrase.

## Measured performance (CPU, 12 logical cores)

| Operation | Latency |
|---|---|
| CLIP model load (cached) | ~0.7 s |
| CLIP concept scoring, cold (text embeddings uncached) | ~3.0 s |
| CLIP concept scoring, warm | **~0.06 s** |
| CLIP full 24-concept vocabulary | ~0.6 s |
| OWL-ViT model load (cached) | ~0.4 s |
| OWL-ViT detection | ~0.8–1.2 s |

Concept text embeddings depend only on the vocabulary, never on the image, so they are
computed once and cached. That is what takes warm scoring from seconds to milliseconds.

Reproduce with `python scripts/check_models.py`.

## Status: why `degraded` and not `available`

These models run real inference and return real scores, but:

- the VQA/captioning answer strategy is still **zero-shot** (contrastive scoring against a
  concept vocabulary, not a head trained for these tasks) — LoRA adaptation improved the
  embedding space, it did not add a task head;
- their softmax outputs are **uncalibrated** — a 56 % score is not a 56 % chance of being
  correct, only a relative ranking within the model's vocabulary;
- CLIP has **no counting or spatial-reasoning** mechanism;
- OWL-ViT is trained on **natural photographs**, so it transfers to discrete overhead
  objects (buildings, aircraft, ships, storage tanks) but is weak on diffuse land cover;
- the Phase 5 adaptation is measured on **one country and one season**, and no user-facing
  task accuracy has been benchmarked yet (Phase 9).

`degraded` is the honest registry status for that: usable, real, and limited. It becomes
`available` only when a capability's accuracy has been established.

## How limitations reach the user

Rather than living only in documentation, each limitation is attached to the results it
affects:

- every VQA/captioning result carries the uncalibrated-score warning;
- grounding results carry the natural-imagery transfer warning;
- captions state they are **retrieval-composed**, not generated;
- coarse regions state they are **tile-level estimates**, not object outlines;
- counting questions are **refused** with an explanation instead of answered with a number;
- if weights cannot be loaded, the result is `unavailable` with no answer invented.

## Methods

**VQA** — the image is scored against a remote-sensing concept vocabulary (24 concepts,
each expanded over 4 prompt templates and averaged). Question type determines the answer
strategy: presence questions report the target concept's probability and rank; "what
dominates" returns the top land-cover concept; open questions list the most likely content.
Counting is refused.

**Captioning** — retrieval-based composition. Concepts clearing a probability threshold are
assembled into a description with their measured scores. Not generative.

**Grounding** — OWL-ViT open-vocabulary detection produces boxes, which are drawn onto an
overlay image and mapped back through the decimation factor and the raster's affine
transform to real coordinates. When the detector returns nothing, a 4×4 tile grid is scored
with CLIP and the best-contrast tiles are merged into a **coarse region**, clearly labelled.

## Planned (Phase 6+)

| Capability | Plan |
|---|---|
| Temporal change, Change VQA | Phase 6 |
| Optical-SAR fusion | Phase 7 — genuine multimodal integration, not concatenated text |
| Benchmark accuracy for VQA/captioning/grounding | Phase 9 |

Training is kept separable from serving: nothing in the backend imports the training module
except the adapter loader, so training can move to a cloud GPU without changing the API.
