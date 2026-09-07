"""Tiny GPU smoke training: prove every stage of the Phase 5 pipeline actually works.

    python scripts/phase5_smoke_train.py

Runs a deliberately tiny adaptation (a handful of samples, 1 epoch) and asserts each stage:

  1. dataset loading            8. backward pass
  2. LMDB image decode          9. optimizer step (parameters actually change)
  3. processor loading         10. adapter save
  4. base model loading        11. adapter reload in a fresh model
  5. LoRA attachment           12. adapted inference
  6. GPU placement             13. adapter differs from base numerically
  7. forward pass + loss

Exits non-zero on the first failure, so a broken pipeline can never be mistaken for a
successful run. Writes ``scripts/phase5_smoke_report.txt``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREPARED = REPO_ROOT / "data" / "bigearthnet" / "prepared"
LMDB_DIR = REPO_ROOT / "data" / "bigearthnet" / "imagery" / "BENv2_lithuania_summer.lmdb"
SMOKE_OUT = REPO_ROOT / "models" / "phase5" / "smoke-adapter"
REPORT = Path(__file__).resolve().parent / "phase5_smoke_report.txt"

STAGES: list[str] = []


def emit(message: str = "") -> None:
    print(message, flush=True)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def stage(number: int, name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    STAGES.append(f"{number:2d}. {name}: {mark}")
    emit(f"[{mark}] {number:2d}. {name}" + (f" - {detail}" if detail else ""))
    return ok


def main() -> int:
    REPORT.unlink(missing_ok=True)
    emit("=" * 74)
    emit("PHASE 5 SMOKE TRAINING (tiny, GPU)")
    emit("=" * 74)

    import torch

    if not torch.cuda.is_available():
        stage(0, "CUDA available", False, "torch.cuda.is_available() is False")
        return 1
    emit(f"  GPU: {torch.cuda.get_device_name(0)}")
    emit(f"  torch {torch.__version__} / CUDA {torch.version.cuda}")
    emit("")

    from ml.training import (
        CaptionDataset,
        LoraTrainingConfig,
        attach_lora,
        build_collate_fn,
        contrastive_loss,
        load_adapted_clip,
        load_split,
        set_seed,
    )

    config = LoraTrainingConfig(
        output_dir=str(SMOKE_OUT),
        epochs=1,
        batch_size=4,
        max_train_samples=8,
        learning_rate=1e-4,
    )
    set_seed(config.seed)

    # 1. dataset loading
    try:
        rows = load_split(PREPARED, "train", config.max_train_samples)
        ok = len(rows) == config.max_train_samples
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(1, "dataset loading", False, repr(exc)) else 1
    if not stage(1, "dataset loading", ok, f"{len(rows)} rows"):
        return 1

    # 2. LMDB image decode
    dataset = CaptionDataset(rows, LMDB_DIR, config.image_size)
    try:
        image, caption = dataset[0]
        ok = image.size == (config.image_size, config.image_size) and bool(caption)
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(2, "LMDB image decode", False, repr(exc)) else 1
    if not stage(2, "LMDB image decode", ok, f"{image.size} RGB, caption {len(caption)} chars"):
        return 1

    from transformers import CLIPModel, CLIPProcessor

    # 3. processor
    try:
        processor = CLIPProcessor.from_pretrained(config.base_model)
        ok = processor is not None
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(3, "processor loading", False, repr(exc)) else 1
    if not stage(3, "processor loading", ok, config.base_model):
        return 1

    # 4. base model
    try:
        model = CLIPModel.from_pretrained(config.base_model)
        base_params = sum(p.numel() for p in model.parameters())
        ok = base_params > 0
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(4, "base model loading", False, repr(exc)) else 1
    if not stage(4, "base model loading", ok, f"{base_params / 1e6:.1f}M params"):
        return 1

    # 5. LoRA attachment
    try:
        model = attach_lora(model, config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        ok = 0 < trainable < total
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(5, "LoRA attachment", False, repr(exc)) else 1
    if not stage(
        5,
        "LoRA attachment",
        ok,
        f"{trainable:,} trainable / {total:,} total ({100 * trainable / total:.3f}%)",
    ):
        return 1

    # 6. GPU placement
    try:
        model.to("cuda")
        device_of_params = {p.device.type for p in model.parameters()}
        ok = device_of_params == {"cuda"}
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(6, "GPU placement", False, repr(exc)) else 1
    if not stage(6, "GPU placement", ok, f"parameter devices: {device_of_params}"):
        return 1

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(processor, config.max_text_tokens),
        drop_last=True,
    )
    batch = next(iter(loader))
    batch = {key: value.to("cuda") for key, value in batch.items()}

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    # 7. forward + loss
    model.train()
    try:
        outputs = model(**batch)
        loss = contrastive_loss(outputs.logits_per_image, outputs.logits_per_text)
        ok = torch.isfinite(loss).item() and loss.item() > 0
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(7, "forward pass + loss", False, repr(exc)) else 1
    if not stage(
        7,
        "forward pass + loss",
        ok,
        f"logits {tuple(outputs.logits_per_image.shape)} on {outputs.logits_per_image.device}, "
        f"loss={loss.item():.6f}",
    ):
        return 1

    # 8. backward
    try:
        loss.backward()
        grad_norm = sum(
            p.grad.norm().item()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        )
        ok = grad_norm > 0
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(8, "backward pass", False, repr(exc)) else 1
    if not stage(8, "backward pass", ok, f"trainable grad norm={grad_norm:.6f}"):
        return 1

    # 9. optimizer step
    try:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
        )
        optimizer.step()
        torch.cuda.synchronize()
        changed = sum(
            1
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not torch.equal(before[name], parameter.detach())
        )
        ok = changed > 0
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(9, "optimizer step", False, repr(exc)) else 1
    if not stage(9, "optimizer step", ok, f"{changed}/{len(before)} LoRA tensors updated"):
        return 1

    peak = torch.cuda.max_memory_allocated() / 1024**2
    emit(f"     peak VRAM during smoke step: {peak:.1f} MB")

    # 10. adapter save
    if SMOKE_OUT.exists():
        shutil.rmtree(SMOKE_OUT)
    try:
        SMOKE_OUT.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(SMOKE_OUT))
        processor.save_pretrained(str(SMOKE_OUT))
        weight_files = list(SMOKE_OUT.glob("adapter_model*"))
        total_size = sum(p.stat().st_size for p in SMOKE_OUT.rglob("*") if p.is_file())
        ok = bool(weight_files) and total_size > 0
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(10, "adapter save", False, repr(exc)) else 1
    if not stage(
        10,
        "adapter save",
        ok,
        f"{[p.name for p in weight_files]}, {total_size / 1024:.0f} KB total",
    ):
        return 1

    # free the training model before reloading
    del model, outputs, loss, batch
    torch.cuda.empty_cache()

    # 11. adapter reload in a fresh model
    try:
        reloaded, reloaded_processor = load_adapted_clip(SMOKE_OUT, device="cuda")
        ok = reloaded is not None and reloaded_processor is not None
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(11, "adapter reload", False, repr(exc)) else 1
    if not stage(11, "adapter reload", ok, "PeftModel loaded from disk"):
        return 1

    # 12. adapted inference
    try:
        image, _caption = dataset[0]
        inputs = reloaded_processor(
            text=["an aerial photograph of forest", "an aerial photograph of water"],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_text_tokens,
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.no_grad():
            adapted_logits = reloaded(**inputs).logits_per_image
            probabilities = adapted_logits.softmax(dim=-1)[0].tolist()
        ok = len(probabilities) == 2 and abs(sum(probabilities) - 1.0) < 1e-3
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(12, "adapted inference", False, repr(exc)) else 1
    if not stage(
        12, "adapted inference", ok, f"probs={[round(p, 4) for p in probabilities]}"
    ):
        return 1

    # 13. the adapter actually changes the model's output
    try:
        base = CLIPModel.from_pretrained(config.base_model).to("cuda").eval()
        with torch.no_grad():
            base_logits = base(**inputs).logits_per_image
        difference = (adapted_logits - base_logits).abs().max().item()
        ok = difference > 0
    except Exception as exc:  # noqa: BLE001
        return 1 if not stage(13, "adapter changes output", False, repr(exc)) else 1
    if not stage(
        13,
        "adapter changes output vs base",
        ok,
        f"max |adapted - base| logit difference = {difference:.6f}",
    ):
        return 1

    emit("")
    emit("=" * 74)
    emit("SMOKE SUMMARY")
    emit("=" * 74)
    for line in STAGES:
        emit(f"  {line}")
    emit("")
    emit("RESULT: SMOKE TRAINING PASSED - the full pipeline runs on GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
