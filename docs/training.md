# Training — remote-sensing VLM adaptation (Phase 5)

This documents the **one adaptation run that actually happened**. Every number here was
measured on this machine and can be re-read from the artifacts the run wrote:

| Artifact | Contents |
|---|---|
| `reports/phase5/training_config.json` | Exact hyperparameters used |
| `reports/phase5/training_metrics.json` | Per-step and per-epoch losses, timing, VRAM |
| `reports/phase5/environment.json` | Versions, GPU, seed, dataset manifest reference |
| `scripts/phase5_gpu_report.txt` | GPU verification before training |
| `scripts/phase5_smoke_report.txt` | 13-stage smoke test of the pipeline |
| `scripts/phase5_training_report.txt` | Human-readable training log |

Evaluation of the result lives in `docs/evaluation.md`. Data provenance lives in
`docs/datasets.md`.

---

## What was done, in one paragraph

`flax-community/clip-rsicd-v2` (CLIP ViT-B/32, 151 M parameters) was adapted with **LoRA**
on **8,362 real BigEarthNet.txt image–caption pairs** using the **symmetric InfoNCE
contrastive objective** CLIP itself is trained with. 1,966,080 of 153,243,393 parameters
(**1.283 %**) were trainable. Training ran for 6 epochs on a real **NVIDIA GeForce RTX 2050
(4 GB)** in **7.6 minutes**, peaking at **2,219.2 MB** of VRAM. Train loss fell
**2.4049 → 1.2647**, validation loss **2.5688 → 2.3774**. The resulting **12.3 MB adapter**
is on disk, reloads in a fresh process, and is served by the live API through the model
registry.

---

## Why this model and this method

Three constraints decided this, and they were checked before any code was written rather
than assumed.

**Constraint 1: 4 GB of VRAM, measured.** `scripts/verify_phase5_gpu.py` confirmed a real
CUDA device with **4.00 GB** total VRAM, compute capability 8.6, driver 555.99, CUDA 12.4.
A 2–3 B parameter generative VLM (Qwen2-VL-2B, PaliGemma-3B) does not fit in 4 GB even under
4-bit QLoRA once activations, optimizer state, and the vision tower are accounted for. CLIP
ViT-B/32 with LoRA fits with headroom — the run used 2.2 GB of 4.0 GB.

**Constraint 2: adaptation has to improve the live product.** The Phase 4 VQA and captioning
specialists are *built on* CLIP image/text embeddings. Adapting CLIP therefore improves what
already ships. Adding a separate generative VLM would have created a second model that
nothing in the serving path uses — an adaptation on paper only.

**Constraint 3: a generative captioner would produce fluent, unverifiable text.** On
satellite imagery a small generative model reliably produces confident and wrong
descriptions. That directly conflicts with this project's no-fabrication rule. A contrastive
dual encoder can only ever report *similarity to a stated concept*, which is a claim the
system can show evidence for.

Rejected alternatives, recorded so the decision is auditable:

| Option | Why rejected |
|---|---|
| Qwen2-VL-2B / PaliGemma-3B + QLoRA | Will not fit 4 GB VRAM; unverified by measurement |
| Train a generative captioner | Fluent-but-wrong output on satellite imagery; violates no-fabrication rule |
| Full fine-tune of CLIP | 151 M trainable params; unnecessary and less reproducible than LoRA |
| Train on `binary` / `mcq` / `bounding box` rows | Wrong objective for a dual encoder (see `docs/datasets.md`) |
| CPU training | Would misrepresent Phase 5; the trainer **raises** rather than silently using CPU |

## The objective

Symmetric InfoNCE over each batch's similarity matrix — the same loss that trains CLIP:

```python
targets = torch.arange(batch_size)
loss = (cross_entropy(logits_per_image, targets)
        + cross_entropy(logits_per_text, targets)) / 2
```

The diagonal holds the true image–caption pairs; every off-diagonal entry is an in-batch
negative. `drop_last=True` keeps the number of negatives constant across steps, which matters
because the loss scale depends on batch size.

LoRA targets `q_proj`, `k_proj`, `v_proj`, `out_proj`. CLIP's vision and text encoders use the
same attention projection names, so one target list adapts **both towers** together — which
is what a contrastive objective requires.

### A real PEFT pitfall, recorded

`LoraConfig(task_type="FEATURE_EXTRACTION")` looks correct for an encoder and fails:

```
TypeError: CLIPModel.forward() got an unexpected keyword argument 'inputs_embeds'
```

PEFT's task-specific wrappers inject arguments that `CLIPModel.forward()` does not accept.
The fix is to omit `task_type` entirely; the generic `PeftModel` forwards kwargs straight
through, which is what a dual encoder needs. The reason is written into
`ml/training/lora_clip.py` so it is not "fixed" back.

## Hyperparameters (exactly as run)

| Parameter | Value |
|---|---|
| Base model | `flax-community/clip-rsicd-v2` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| Trainable / total parameters | 1,966,080 / 153,243,393 (**1.283 %**) |
| Epochs | 6 |
| Batch size | 32 (no gradient accumulation) |
| Optimizer | AdamW, lr `1e-4`, weight decay 0.01 |
| Schedule | 10 % linear warmup → cosine decay to 0 |
| Gradient clipping | 1.0 (max-norm) |
| Precision | fp16 autocast + `GradScaler` |
| Seed | 42 (Python, NumPy, Torch, CUDA) |
| Image size | 224×224 |
| Max text tokens | 77 (CLIP's fixed limit) |
| Optimizer steps | 750 |

Batch size is **probed downward on CUDA OOM** by `scripts/train_phase5.py`, so the run adapts
to whatever VRAM is free instead of failing. On this machine 32 succeeded on the first
attempt.

## Measured results

| Epoch | Train loss | Validation loss |
|---|---|---|
| 1 | 2.4049 | 2.5688 |
| 2 | 1.7954 | 2.4123 |
| 3 | 1.5826 | 2.4098 |
| 4 | 1.4422 | 2.4307 |
| 5 | 1.3383 | 2.3833 |
| 6 | **1.2647** | **2.3774** |

| | |
|---|---|
| Duration | **456.0 s (7.6 min)** |
| Peak VRAM allocated | **2,219.2 MB** of 4,096 MB |
| Best validation loss | 2.3774 (epoch 6) |
| Adapter weights | `adapter_model.safetensors`, 7,892,256 bytes |
| Adapter directory total | 12,929,782 bytes (12.3 MiB) — weights + tokenizer + processor config |

**Reading these losses honestly.** Train loss fell 47 % while validation loss fell only 7.4 %
(2.5688 → 2.3774), and epoch 4 regressed (2.4307) before recovering. That gap is the
signature of a model beginning to fit 4,008 training samples. Validation loss was still
improving at epoch 6, so training was not stopped early for overfitting — but the trend says
more epochs at this dataset size would need regularisation or more data, not just more steps.
The loss values themselves are batch-size-dependent contrastive losses and are **not**
comparable across runs with different batch sizes; the held-out retrieval metrics in
`docs/evaluation.md` are the meaningful measure of whether adaptation worked.

## GPU verification (a hard gate before training)

Phase 5 is specified as GPU adaptation, so `train_lora_clip` **raises** if
`torch.cuda.is_available()` is false rather than falling back to CPU. Before training,
`scripts/verify_phase5_gpu.py` proved real GPU execution end to end and exits non-zero
otherwise:

| Check | Result |
|---|---|
| Device | NVIDIA GeForce RTX 2050, 4.00 GB, CC 8.6, driver 555.99 |
| Stack | torch 2.6.0+cu124 (CUDA 12.4), transformers 4.57.6, peft 0.20.0, accelerate 1.14.0 |
| CUDA matmul vs CPU | max abs difference 6.10e-05 |
| Forward / backward / optimizer step | loss 1.034706, grad norm 0.483285, 4/4 tensors updated |
| fp16 autocast step | OK (loss 0.936465) |

> **Hardware note.** Windows also reports an "NVIDIA GeForce RTX 3050 6GB" on this machine,
> but it is a phantom device in error state (Code 45) and is not usable. The RTX 2050 is the
> real GPU. Recorded because a 6 GB figure would otherwise look available and is not.

## Smoke test before the real run

`scripts/phase5_smoke_train.py` runs 13 staged checks on a handful of samples and exits
non-zero on the first failure, so a broken pipeline cannot be mistaken for a successful run.
It covers: dataset load → LMDB read → RGB conversion → processor → LoRA attach → forward →
contrastive loss → backward → optimizer step → adapter save → adapter reload → adapted
inference → **and that the adapter actually changes the output** (measured max logit delta
0.672132 vs base; peak VRAM during the smoke step 894.2 MB). That last check is the one that
matters: it distinguishes a trained adapter from a saved no-op.

## Reproduce

```powershell
$env:HF_HOME="D:\SatQuery-AI\models\hf-cache"

# CUDA build of torch (CPU-only dev uses .../whl/cpu instead)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[training]"

# 0. Hard gate: prove the GPU really works
.\.venv\Scripts\python.exe scripts\verify_phase5_gpu.py

# 1. Data (see docs/datasets.md for the full sequence)
.\.venv\Scripts\python.exe scripts\download_bigearthnet.py
.\.venv\Scripts\python.exe scripts\prepare_bigearthnet_txt.py

# 2. Smoke test the whole pipeline on a few samples
.\.venv\Scripts\python.exe scripts\phase5_smoke_train.py

# 3. Real training (7.6 min on an RTX 2050)
.\.venv\Scripts\python.exe scripts\train_phase5.py --epochs 6 --batch-size 32

# 4. Baseline vs adapted on the held-out test split
.\.venv\Scripts\python.exe scripts\evaluate_phase5.py --split test

# 5. End-to-end verification through the real API
.\.venv\Scripts\python.exe scripts\verify_phase5.py
```

### Reproducibility caveats

- **Seed 42** is set for Python, NumPy, Torch, and CUDA. cuDNN autotuning and fp16 reductions
  are not forced deterministic, so losses reproduce closely but not bit-exactly.
- `reports/phase5/environment.json` records `git_commit: "HEAD"` because **the repository has
  no commits yet**. That is a real gap: the run cannot be pinned to a revision. It resolves
  itself on the first commit; the script reads the real SHA when one exists.
- The dataset is pinned by SHA256 in `data/manifests/bigearthnet_txt_manifest.json`, so the
  data side of the run is verifiable even without a commit.
- Long-running Python output is written to files with `flush()` + `os.fsync()` because this
  shell truncates stdout from long-running processes. The report files, not the terminal
  scrollback, are the reliable record.

## Serving the adapter

Training is separable from serving — nothing in the backend imports the training module
except the loader. At serving time:

- `PHASE5_ADAPTER_DIR` (default `models/phase5/adapter`) locates the adapter. Absolute paths
  are required in Docker, where `ml/` lives in site-packages.
- If the directory is missing or has no weight file, `resolve_variant()` returns the
  **baseline** and the execution trace records *why* it fell back. A downgrade is never
  silent and the adapted model is never claimed when it did not load.
- The registry exposes `vqa`/`captioning` (baseline) and `vqa_adapted`/`captioning_adapted`
  (adapted) as **separate entries**, so a plan and its trace state unambiguously which model
  ran. A runtime `variant` parameter was rejected for exactly that reason.
- `peft` is required to load the adapter and is in both the `inference` extra and the backend
  Docker image; without it the container would silently serve baseline only.

**Grounding was deliberately not adapted.** No grounding-specific training or evaluation was
done, so no grounding improvement is claimed. `ADAPTED_PREFERENCE` has no `grounding` key and
a test asserts that it does not.

## Not done in Phase 5

Stated plainly so nothing here is over-read:

- **No optical–SAR fusion was learned.** Sentinel-1 is present in the data and unused. Phase 7.
- **No temporal/change training.** Phase 6.
- **No generative captioning.** Captions remain retrieval-composed from concept scores.
- **No benchmark evaluation** on standard remote-sensing benchmarks (RSVQA, LEVIR-CC). The
  Phase 5 evaluation is retrieval on the dataset's own held-out split. Phase 9.
- **No calibration.** Scores remain uncalibrated, which is why the adapted registry entries
  are `degraded`, not `available`.
- **No multi-country or multi-season training.** One country, one season, one sensor.
