# change_model — Capability Matrix

Determined from actual configuration, weight structure, and **real inference on real
remote-sensing imagery** — never from the folder name. The checkpoint is
`BlipForQuestionAnswering` (BLIP ViT-B/16 + BERT text encoder/decoder), 361,230,140 params,
single 384×384 3-channel image input, generative text output.

| Capability | Supported | Evidence |
|---|---|---|
| Single-image VQA (generative) | **YES** | `BlipForQuestionAnswering`; real inference produces text answers (e.g. "trees", "yes", "0_to_10") on SECOND/demo images. |
| Change VQA (CDVQA answer space) | **YES (weak)** | Emits CDVQA-specific tokens (`0_to_10`, `nvg_surface`); held-out CDVQA Test **0.5467** / Test2 **0.5033** — below the existing `cdvqa_trained` (0.7114/0.6632) and barely above majority (0.5084). |
| Temporal / bi-temporal reasoning | **NO** | Single-image architecture (`patch_embedding` = `[768, 3, 16, 16]`, 3 channels). t1 / t2 / side-by-side conventions gave **identical** val accuracy (0.60) and identical predictions → the pair is not used. |
| Captioning (descriptive) | **NO (not as-is)** | VQA head, not a captioner; fine-tuned to CDVQA short answers, so it outputs answer tokens, not scene descriptions. |
| Grounding / localization | **NO** | No detection/box head; no localization tensors. |
| Binary change detection | **NO** | No segmentation/mask head; generative text only. Cannot output a change mask. |
| Semantic change detection (from→to) | **NO** | Single-image, no segmentation head, no temporal input. |
| Change captioning | **NO** | Same as captioning; not trained/структured for it. |
| Optical + SAR fusion | **NO** | Single RGB image; no multimodal fusion path. |

## Bottom line
A **single-image BLIP VQA** checkpoint fine-tuned on the CDVQA answer vocabulary. It is not a
change-detection model, not bi-temporal, not a segmenter, and not a general RS captioner.
