# Phase 14 — Dataset Matrix

**Project:** SatQuery AI (SIH26167)
**Generated:** 2026-09-07

Records what is on disk vs what each Phase 14 capability would need, with provenance/license and provisioning feasibility in **this** environment (HF ~0.02 MB/s; non-HF hosts DNS-restricted; 4 GB VRAM).

## Present locally

| Dataset | Location | Size | Content | License / provenance | Used by |
|---|---|---|---|---|---|
| SECOND | `D:\dataset\second\{im1,im2,label...}` | 2.3 GB (11,872 files) | Bi-temporal aerial RGB (0.1524 m) + semantic labels | SECOND, research use | learned SECOND, CDVQA (imagery), Phase 14F |
| LEVIR-CD (test) | `D:\dataset\levir_cd\test.parquet` | 66 MB | Bi-temporal building-change tiles 256px | `ericyu/LEVIRCD_Cropped_256` (HF) | Phase 11 cross-dataset eval |
| OSCD | `D:\dataset\oscd\*.parquet` | 1 MB | Sentinel-2 change tiles 256px | `ericyu/OSCD_Cropped_256` (HF) | Phase 11 cross-dataset eval |
| VRSBench (annotations) | `D:\dataset\vrsbench\referring.json` | 9.8 MB | 16,159 referring expressions / 9,318 images | VRSBench (`xiang709/VRSBench`) | grounding eval (images missing) |
| Demo fixtures | `data/demo/*.tif` | ~0.7 MB | Synthetic GeoTIFF (optical/SAR/T1/T2/incompatible) | CC0 synthetic (this project) | E2E demo/tests |
| Model weights | `models/hf-cache`, `models/phase5`, `models/phase9`, `change_model/` | ~ GBs | clip-rsicd, owlvit, LoRA adapter, SECOND/CDVQA ckpts, BLIP | see model cards / MIT (project) | production/registry |

## Needed but NOT present (and provisioning feasibility)

| Capability | Needed dataset | Approx size | Feasible here? | Reason |
|---|---|---|---|---|
| 14A Real VQA | RSVQA (LR/HR) or VRSBench-VQA | ~GBs (Sentinel/aerial tiles) | **NO — BLOCKED** | Not local; HF download ~0.02 MB/s → many hours–days |
| 14B Generative captioning | VRSBench-Cap / RSICD / RSICD-caption | RSICD ~ hundreds MB; VRSBench images 3.79 GB | **NO — BLOCKED** | Not local; download infeasible at measured rate |
| 14C Land-cover segmentation | e.g. LoveDA / Potsdam / DeepGlobe | multi-GB | **NO — BLOCKED** | Not local; download infeasible |
| 14D Detector / counting | e.g. DOTA / DIOR / NWPU-VHR-10 | multi-GB | **NO — BLOCKED** | Not local; download infeasible |
| 14E Grounding IoU | VRSBench `Images_val.zip` | 3.79 GB | **NO — BLOCKED** | Measured ~0.02 MB/s → ~50 h |
| 14F Temporal improvement | SECOND (local) | 2.3 GB | **YES** | Already local; small model fits 4 GB VRAM |
| 14G Semantic from-to change | SECOND semantic labels (local) | included | **PARTIAL** | Labels local, but a real from-to model + eval is heavy on 4 GB VRAM |
| 14H Optical+SAR benchmark | Labeled co-registered optical+SAR (e.g. SEN1-2 / WHU-OPT-SAR) | multi-GB | **NO — BLOCKED** | Not local; download infeasible; current fusion is deterministic (no labels needed) |

## Dataset-rule compliance

- Only public/open datasets are used or proposed; provenance/license recorded above.
- **No hidden ISRO/SAC or challenge data** anywhere.
- No benchmark annotations are used during training or leaked into inference.
- Demo data is synthetic (`data/manifests/demo_manifest.json`, `hidden_evaluation_data: false`).
