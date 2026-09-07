"""Inference for the external `caption_model/` BLIP captioning checkpoint.

The checkpoint at ``caption_model/`` is a **BlipForConditionalGeneration** model fine-tuned
on VRSBench captions. It is a *generative* captioner: given one image it generates a free-text
description, word by word — unlike the project's existing CLIP-based captioning specialist,
which composes text from concept scores rather than generating it.

Weights are loaded read-only. If the checkpoint or its dependencies are absent,
``availability()`` reports it and the specialist degrades gracefully instead of guessing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .base import (
    Availability,
    ModelUnavailableError,
    load_cached,
    resolve_device,
    torch_available,
    transformers_available,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    """Directory holding the external BLIP checkpoint (configurable, defaults to caption_model/)."""
    configured = os.environ.get("CAPTION_MODEL_DIR", "caption_model")
    path = Path(configured)
    return path if path.is_absolute() else _repo_root() / path


def availability() -> Availability:
    """Whether the external BLIP captioning checkpoint can run right now."""
    if not torch_available():
        return Availability(False, "PyTorch is not installed.")
    if not transformers_available():
        return Availability(False, "Transformers is not installed.")
    d = model_dir()
    needed = ["config.json", "model.safetensors"]
    missing = [f for f in needed if not (d / f).is_file()]
    if missing:
        return Availability(
            False, f"caption_model checkpoint incomplete (missing {', '.join(missing)})."
        )
    return Availability(True)


@dataclass
class CaptionModelPrediction:
    """One generated caption from the BLIP captioning checkpoint."""

    caption: str
    raw: str
    #: Uncalibrated mean per-token generation probability (a real model signal, not a
    #: calibrated probability of correctness). ``None`` if scores were unavailable.
    confidence: float | None
    model_version: str
    device: str


def _load():
    import torch  # noqa: F401
    from transformers import BlipForConditionalGeneration, BlipProcessor

    status = availability()
    if not status.available:
        raise ModelUnavailableError(status.reason)

    d = model_dir()
    device = resolve_device()
    try:
        processor = BlipProcessor.from_pretrained(d, use_fast=True)
    except Exception:  # noqa: BLE001 - fall back to the slow processor
        processor = BlipProcessor.from_pretrained(d)
    model = BlipForConditionalGeneration.from_pretrained(d).eval().to(device)
    return model, processor, device


@lru_cache(maxsize=1)
def _cached():
    return load_cached("caption-model-blip", _load)


def model_version() -> str:
    return "blip-captioning-vrsbench-external (single-image, generative)"


def predict(image, max_new_tokens: int = 50) -> CaptionModelPrediction:
    """Generate a caption for a single RGB image with the BLIP checkpoint."""
    import torch

    model, processor, device = _cached()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
    seq = out.sequences[0]
    raw = processor.decode(seq, skip_special_tokens=True)
    caption = raw.strip()

    confidence: float | None = None
    if getattr(out, "scores", None):
        probs = []
        gen_tokens = seq[-len(out.scores):]
        for logits, token_id in zip(out.scores, gen_tokens, strict=False):
            p = torch.softmax(logits[0], dim=-1)[int(token_id)].item()
            probs.append(p)
        if probs:
            confidence = float(sum(probs) / len(probs))

    return CaptionModelPrediction(
        caption=caption,
        raw=raw,
        confidence=confidence,
        model_version=model_version(),
        device=device,
    )
