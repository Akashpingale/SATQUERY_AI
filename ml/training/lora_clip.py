"""LoRA adaptation of CLIP on BigEarthNet.txt image-caption pairs.

Method: parameter-efficient fine-tuning of a CLIP dual encoder with the standard symmetric
InfoNCE (contrastive) objective, so the image tower and the text tower are pulled into
agreement on remote-sensing content.

Why CLIP and not a generative VLM: the GPU available is a 4 GB RTX 2050. CLIP ViT-B/32
(~151 M parameters) trains comfortably there with LoRA, whereas a 2-3 B generative VLM would
need aggressive quantisation and still be memory-bound. Just as importantly, the Phase 4
VQA and captioning specialists are *built on* CLIP embeddings, so adapting CLIP improves the
live product immediately rather than adding a parallel model that nothing uses.

Everything here is real training: real forward passes, a real contrastive loss, real
gradients, and a real PEFT adapter written to disk.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Attention projections in both towers. CLIP's vision and text encoders both use these
# names, so one target list adapts image and text representations together.
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj")

CLIP_MAX_TEXT_TOKENS = 77


@dataclass
class LoraTrainingConfig:
    """Everything needed to reproduce an adaptation run."""

    base_model: str = "flax-community/clip-rsicd-v2"
    output_dir: str = "models/phase5/adapter"

    # LoRA
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    # Optimisation
    epochs: int = 3
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Runtime
    precision: str = "fp16"  # fp16 | fp32
    seed: int = 42
    num_workers: int = 0
    image_size: int = 224
    max_text_tokens: int = CLIP_MAX_TEXT_TOKENS
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    log_every: int = 10

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload


@dataclass
class TrainingMetrics:
    """Measured metrics from a run. Never synthesised."""

    steps: list[dict[str, float]] = field(default_factory=list)
    epochs: list[dict[str, float]] = field(default_factory=list)
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    best_val_loss: float | None = None
    train_seconds: float = 0.0
    peak_vram_bytes: int | None = None
    trainable_parameters: int = 0
    total_parameters: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(prepared_dir: str | Path, split: str, limit: int | None = None) -> list[dict]:
    """Load a prepared split written by ``scripts/prepare_bigearthnet_txt.py``."""
    path = Path(prepared_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"prepared split not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if limit is not None:
        rows = rows[:limit]
    return rows


class CaptionDataset:
    """Torch dataset yielding ``(PIL image, caption)`` from the LMDB + prepared split.

    Implemented against ``torch.utils.data.Dataset`` at runtime without importing torch at
    module import time, so this module stays importable when torch is absent.
    """

    def __init__(self, rows: list[dict], lmdb_dir: str | Path, image_size: int = 224) -> None:
        from ml.datasets import LmdbImageReader

        self.rows = rows
        self.reader = LmdbImageReader(lmdb_dir)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows[index]
        rgb = self.reader.read_rgb(row["patch_id"])
        if rgb is None:
            # Preparation verified every key, so this indicates real corruption rather than
            # an expected miss. Fail loudly instead of substituting a blank image.
            raise RuntimeError(f"image entry unreadable at train time: {row['patch_id']}")
        image = Image.fromarray(rgb, "RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        return image, row["caption"]


def build_collate_fn(processor, max_text_tokens: int = CLIP_MAX_TEXT_TOKENS):
    """Build a collate function turning ``(image, caption)`` pairs into model inputs.

    Captions are truncated to CLIP's fixed 77-token context. This is a severe constraint on
    this dataset, and the real numbers are measured rather than assumed
    (``scripts/measure_caption_tokens.py``, over all 8,362 prepared samples):

    - median caption length 200 tokens, mean 188.5, max 449;
    - **84.0 % of captions exceed 77 tokens**;
    - on average only **49.1 %** of a caption's tokens reach the model.

    The retained prefix is the caption's opening summary plus its dominant land cover, which
    is the part that describes the image as a whole; the discarded tail is per-parcel area
    and adjacency detail. That makes the truncation tolerable rather than harmless, and it is
    a documented ceiling on what this adaptation can learn - see ``docs/datasets.md``.
    """

    def collate(batch):
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        encoded = processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_tokens,
        )
        return encoded

    return collate


def attach_lora(model, config: LoraTrainingConfig):
    """Wrap a CLIP model with LoRA adapters and report trainable parameter counts."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias="none",
        # Deliberately no task_type. CLIP is a dual encoder, and PEFT's task-specific
        # wrappers (including FEATURE_EXTRACTION) inject arguments such as `inputs_embeds`
        # that CLIPModel.forward() does not accept. The generic PeftModel forwards kwargs
        # straight through to the base model, which is what a dual encoder needs.
    )
    return get_peft_model(model, lora_config)


def contrastive_loss(logits_per_image, logits_per_text):
    """Symmetric InfoNCE loss, as used to train CLIP.

    The diagonal of the similarity matrix holds the true image-caption pairs, so the targets
    are ``arange(batch)`` in both directions.
    """
    import torch
    import torch.nn.functional as functional

    batch = logits_per_image.shape[0]
    targets = torch.arange(batch, device=logits_per_image.device)
    image_loss = functional.cross_entropy(logits_per_image, targets)
    text_loss = functional.cross_entropy(logits_per_text, targets)
    return (image_loss + text_loss) / 2.0


def evaluate_loss(model, loader, device: str, precision: str) -> float:
    """Mean contrastive loss over a loader. No gradients."""
    import torch

    model.eval()
    total = 0.0
    batches = 0
    autocast_enabled = precision == "fp16" and device == "cuda"

    with torch.no_grad():
        for encoded in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
                outputs = model(**encoded)
                loss = contrastive_loss(outputs.logits_per_image, outputs.logits_per_text)
            total += float(loss.item())
            batches += 1
    return total / max(1, batches)


def train_lora_clip(
    config: LoraTrainingConfig,
    prepared_dir: str | Path,
    lmdb_dir: str | Path,
    *,
    progress=None,
) -> tuple[TrainingMetrics, Path]:
    """Run LoRA adaptation and save the adapter.

    Returns ``(metrics, adapter_path)``. Raises if CUDA is unavailable, because Phase 5 is
    specified as GPU adaptation and silently training on CPU would misrepresent the run.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import CLIPModel, CLIPProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run Phase 5 training on CPU.")

    device = "cuda"
    set_seed(config.seed)

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    report(f"loading base model: {config.base_model}")
    processor = CLIPProcessor.from_pretrained(config.base_model)
    model = CLIPModel.from_pretrained(config.base_model)

    model = attach_lora(model, config)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    report(f"trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    train_rows = load_split(prepared_dir, "train", config.max_train_samples)
    val_rows = load_split(prepared_dir, "validation", config.max_eval_samples)
    report(f"train samples: {len(train_rows):,} | validation samples: {len(val_rows):,}")

    collate = build_collate_fn(processor, config.max_text_tokens)
    train_loader = DataLoader(
        CaptionDataset(train_rows, lmdb_dir, config.image_size),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
        drop_last=True,  # contrastive loss needs a consistent batch of negatives
    )
    val_loader = DataLoader(
        CaptionDataset(val_rows, lmdb_dir, config.image_size),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress_ratio)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metrics = TrainingMetrics(trainable_parameters=trainable, total_parameters=total)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        counted = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, encoded in enumerate(train_loader, start=1):
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(**encoded)
                loss = contrastive_loss(outputs.logits_per_image, outputs.logits_per_text)

            scaled_loss = loss / config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if batch_index % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % config.log_every == 0:
                    entry = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": float(loss.item()),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                    }
                    metrics.steps.append(entry)
                    report(
                        f"  step {global_step}/{total_steps} "
                        f"loss={entry['loss']:.4f} lr={entry['learning_rate']:.2e}"
                    )

            running += float(loss.item())
            counted += 1

        train_loss = running / max(1, counted)
        val_loss = evaluate_loss(model, val_loader, device, config.precision)
        metrics.epochs.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        report(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        metrics.final_train_loss = train_loss
        metrics.final_val_loss = val_loss
        if metrics.best_val_loss is None or val_loss < metrics.best_val_loss:
            metrics.best_val_loss = val_loss

    metrics.train_seconds = time.time() - started
    metrics.peak_vram_bytes = int(torch.cuda.max_memory_allocated())

    adapter_path = Path(config.output_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    report(f"adapter saved to {adapter_path}")

    return metrics, adapter_path


def load_adapted_clip(adapter_dir: str | Path, base_model: str | None = None, device: str = "cpu"):
    """Load a CLIP model with the Phase 5 adapter applied.

    Returns ``(model, processor)``. The base model id is read from the adapter config unless
    overridden, so an adapter always reloads against the base it was trained on.
    """
    from peft import PeftConfig, PeftModel
    from transformers import CLIPModel, CLIPProcessor

    adapter_dir = Path(adapter_dir)
    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    base = base_model or peft_config.base_model_name_or_path

    model = CLIPModel.from_pretrained(base)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    model.to(device)

    try:
        processor = CLIPProcessor.from_pretrained(str(adapter_dir))
    except Exception:  # noqa: BLE001 - fall back to the base processor
        processor = CLIPProcessor.from_pretrained(base)

    return model, processor
