# change_model — Verification Report (Phase 10)

**Final decision: `VERIFIED_BUT_NOT_INTEGRATED`.**

An externally-supplied checkpoint was placed at `D:\SatQuery-AI\change_model\`. It was
forensically inspected, verified, loaded, and evaluated with **real inference on real data**. It
is a single-image BLIP VQA model fine-tuned on the CDVQA answer vocabulary. On the same held-out
CDVQA data it **underperforms the existing `cdvqa_trained` specialist** and it provides **no
capability the project is missing**, so — per the phase's own rule ("if it does not provide a
material benefit, keep it out of production") — it is verified and preserved but not integrated.
The original files were not modified.

## 1. Model identity
`BlipForQuestionAnswering` (`model_type: blip`), config `transformers_version 5.16.1`.
Despite the folder name, it is **not** a change-detection model.

## 2. Architecture
BLIP: `blip_vision_model` (ViT-B/16, image 384, patch 16, 12 layers, hidden 768) +
`blip_text_model` BERT encoder (12 layers) + BERT text **decoder** with an LM head
(`text_decoder.cls.predictions`, vocab 30524). Processor `BlipProcessor`/`BlipImageProcessor`
(384×384, 3-ch, ImageNet-CLIP normalisation); tokenizer `BertTokenizer`.

## 3. Base model
Structurally `Salesforce/blip-vqa-base` (BLIP VQA). No `base_model` field is encoded; identity
inferred from architecture + config.

## 4. Parameter count
**361,230,140** parameters, 787 tensors, all **float32** (→ complete model, not an adapter;
1.445 GB on disk is consistent).

## 5. Weight verification
Valid safetensors; header parsed (`__metadata__ = {"format": "pt"}`). Modules: `vision_model`
(150 tensors), `text_encoder` (316), `text_decoder` (321). Head = generative LM
(`text_decoder.cls.predictions.*`). **No** segmentation/mask/detection/classifier head.
`vision_model.embeddings.patch_embedding.weight = [768, 3, 16, 16]` → **3 input channels, one
image** (not 6-channel / bi-temporal). Inventory: `reports/change_model/checkpoint_inventory.json`.

## 6. SHA256
| File | SHA256 |
|---|---|
| model.safetensors | `e1dd3d496b0472d9d3f9d051810be066f440bdd4178b35d990128b6590469f2f` |
| config.json | `1541d3219159610edb13f7691b426b344ee59de19cf42ec511743ca1d10fc44c` |
| generation_config.json | `c968bc803cfc5245ebc1eb86c8ed51889c028e562812360e145605352e5fd1fa` |
| processor_config.json | `f549e6de94db8ba6018e92408ea9950a6df44721a432e919e29c42464236fb1f` |
| tokenizer.json | `f17212c38517cfedc9306b57eeffbaf58c4976fcb42e5a87b405f36a7b9a9eb2` |
| tokenizer_config.json | `ce6fab2c6fb8634675a4786ff7d8458cdf6eafd864afb7f4edb40f671753dcdf` |

## 7. Training status
**Fine-tuned** (not generic). Evidence: real inference emits **CDVQA-specific answer tokens**
(`0_to_10`, `nvg_surface`) that stock `blip-vqa-base` would never produce, and it scores well
above chance on CDVQA. So it is a BLIP VQA fine-tuned on CDVQA.

## 8. Training provenance
**Not independently verifiable from checkpoint metadata** (`__metadata__` only records
`format: pt`; no dataset/hyperparameters encoded). Behavioural evidence indicates CDVQA
fine-tuning. Per the rules, provenance beyond that is not fabricated: "Training provenance, if
supplied by the trusted project source, is not encoded/verifiable from checkpoint metadata."

## 9. Intended task
Single-image VQA, specialised to the CDVQA answer space.

## 10. Capability matrix
See `CHANGE_MODEL_CAPABILITY_MATRIX.md`. YES: single-image generative VQA; weak CDVQA VQA.
NO: temporal/bi-temporal reasoning, binary change, semantic change, grounding, captioning,
change captioning, optical+SAR.

## 11. Input/output format
Input: one RGB image (384×384) + text question. Output: generated text (max_length 20).

## 12. Hardware requirements
Loads on CPU in ~1.0 s; on RTX 2050 VRAM allocated **1447.5 MB**, peak **1508.5 MB** (fits 4 GB).
Inference latency ~0.2–0.5 s/question (GPU, batched). See `reports/change_model/load_test.json`.

## 13. Load test — **PASS**
Loaded with `transformers` `BlipForQuestionAnswering.from_pretrained` (safetensors, no pickle);
361 M params; moved to CUDA in ~4 s.

## 14. Minimal inference test — **PASS**
Real RS images (SECOND `im1/00020.png`, demo optical GeoTIFF). Sample: "Is there water?" → "no";
"Are there buildings?" → "yes"; "How many roads?" → "0_to_10". Valid, finite text; no NaN/Inf.

## 15. SECOND evaluation — **NOT APPLICABLE**
The model has no segmentation/mask head and takes a single 3-channel image, so it **cannot**
produce a binary change mask. SECOND IoU/F1 cannot be computed. Current SECOND remains:
learned `second_change_learned` IoU 0.4219 / F1 0.5934; deterministic 0.2248 / 0.367.

## 16. Semantic change evaluation — **NOT APPLICABLE**
No semantic-segmentation head, no bi-temporal input. Cannot predict T1→T2 class maps.

## 17. CDVQA evaluation — **DONE (real held-out)**
Pair-presentation convention selected on the **Val** split only (t1/t2/side-by-side all 0.60 →
`t1`), then frozen for Test/Test2 (seeded subsamples; Wilson CIs):

| Split | External BLIP | `cdvqa_trained` (existing) | Majority baseline |
|---|---|---|---|
| Test | **0.5467** (n=600, CI 0.507–0.586) | **0.7114** (full 39,686) | 0.5084 |
| Test2 | **0.5033** (n=600, CI 0.463–0.543) | **0.6632** (full 31,036) | 0.4560 |

The three conventions gave identical predictions → the model does not use temporal information.
Weakest types: `change_to_what` ≈ 0.02, `largest_change` ≈ 0.10. Evidence:
`reports/change_model/cdvqa_evaluation.json`, `cdvqa_{test,test2}_predictions.jsonl`.

## 18. Training / fine-tuning performed
**None.** The model is already CDVQA-fine-tuned but underperforms the existing specialist, and it
is architecturally single-image (mismatched to the *change* task). Further fine-tuning a 361 M
single-image model to at best match an existing 832 K-param classifier that already scores 0.71
is not justified (no material gain, wrong architecture for temporal change). No retraining done.

## 19. Comparison with current models
- vs `cdvqa_trained` (same CDVQA task): **worse by ~16 points** on both Test and Test2; only
  +3.8 pts over majority on Test.
- vs `second_change_learned` / Phase 6 CVA: **not comparable** — the model cannot do change
  detection (no mask head, single image).

## 20. Material-benefit analysis
Checked against all eight material-benefit criteria: (1) outperform existing — **no**;
(2) missing capability — **no** (its generative VQA is overfit to CDVQA answers, emitting
`0_to_10` on generic optical, so not a usable general RS-VQA); (3) better RS reasoning — **no**
(weaker); (4) semantic temporal beyond binary SECOND — **no** (single-image); (5) better
change-VQA — **no** (worse); (6) change captioning — **no**; (7) useful VLM for SIH — **no**
(existing specialist better); (8) improves answer/evidence quality — **no**. → **Not material.**

## 21. Integration decision
**Not integrated.** No registry entry, no specialist, no agent routing, no UI change. The
checkpoint is preserved untouched; existing functionality is unchanged.

## 22–26. Registry / routing / confidence / evidence / UI changes
**None** (not integrated). All Phase 1–9 specialists, routing, confidence, evidence, trace,
report generation and the (recently fixed) image preview are unchanged.

## 27. Regression
No production code (`backend/app`, `frontend/src`, registry, planner) was modified in Phase 10 —
only read-only inspection/eval scripts and reports/docs were added. The Phase 9 baseline
(backend 447, frontend 126, Playwright 6/6, ruff/eslint/tsc/build clean, docker config valid)
therefore stands unchanged; verified by re-running lint and a backend smoke (see report footer).

## 28. Security checks
- Weights loaded only from `change_model/` via **safetensors** (no pickle, no arbitrary code).
- No associated scripts shipped with the model; nothing from the folder was executed.
- No network access required (loaded offline, `HF_HUB_OFFLINE=1`).
- No secrets found in the model files. No hidden-dataset access.

## 29. Known limitations
Single-image only (no temporal); no segmentation; generative output overfit to CDVQA answer
vocabulary; underperforms the existing CDVQA specialist; provenance not encoded in the checkpoint.

## 30. Reproduction
```
python scripts/inspect_change_model.py                       # SHA256 + safetensors header
python scripts/load_test_change_model.py                     # load + minimal real inference
python scripts/eval_change_model_cdvqa.py --val-samples 150 --test-samples 600 --batch-size 16
```
Place the model at `D:\SatQuery-AI\change_model\` (gitignored; not committed). Requires the
CDVQA JSON (`datasets/cdvqa/`) and SECOND imagery (`D:\dataset\second`) provisioned in Phase 9.
