"""Inference for the external `change_model/` BLIP Change-VQA checkpoint.

The checkpoint at ``change_model/`` is a **BlipForQuestionAnswering** model (single image,
384px, 3 channels) fine-tuned on CDVQA. It is a *generative* VQA model: given one image and
a question it generates a short free-text answer. It is NOT a two-image temporal detector.

For a bi-temporal (T1/T2) question a single-image model must be shown one image. The
pair-presentation convention was selected on the CDVQA **validation** split only and then
frozen for the held-out Test/Test2 evaluation (see ``reports/change_model``). That frozen
choice was ``t1``; it is reused here so production behaviour matches the benchmark. The
convention is never re-selected on test data.

Weights are loaded read-only. If the checkpoint or its dependencies are absent,
``availability()`` reports it and the specialist degrades gracefully instead of guessing.
"""

from __future__ import annotations

import os
import re
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

#: Pair-presentation convention frozen from CDVQA validation (reports/change_model). Never
#: re-selected on Test/Test2. Overridable only for reproducing the validation sweep.
FROZEN_CONVENTION = "t1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    """Directory holding the external BLIP checkpoint (configurable, defaults to change_model/)."""
    configured = os.environ.get("CHANGE_MODEL_DIR", "change_model")
    path = Path(configured)
    return path if path.is_absolute() else _repo_root() / path


def availability() -> Availability:
    """Whether the external BLIP Change-VQA checkpoint can run right now."""
    if not torch_available():
        return Availability(False, "PyTorch is not installed.")
    if not transformers_available():
        return Availability(False, "Transformers is not installed.")
    d = model_dir()
    needed = ["config.json", "model.safetensors"]
    missing = [f for f in needed if not (d / f).is_file()]
    if missing:
        return Availability(
            False, f"change_model checkpoint incomplete (missing {', '.join(missing)})."
        )
    return Availability(True)


@dataclass
class ChangeModelPrediction:
    """One generated answer from the BLIP Change-VQA checkpoint."""

    answer: str
    raw: str
    #: Uncalibrated mean per-token generation probability (a real model signal, not a
    #: calibrated probability of correctness). ``None`` if scores were unavailable.
    confidence: float | None
    convention: str
    model_version: str
    device: str


def _normalize_answer(text: str) -> str:
    """Normalise a generated answer the same way the frozen CDVQA evaluation did.

    The BERT word-piece decoder renders CDVQA underscore tokens with spaces
    (``"0 _ to _ 10"`` for ``"0_to_10"``); spaces around underscores are collapsed. Only
    formatting is normalised - no synonym mapping - so nothing is inflated.
    """
    t = text.strip().lower()
    t = re.sub(r"\s*_\s*", "_", t)
    t = " ".join(t.split())
    return t.rstrip(".!?")


def _load():
    import torch  # noqa: F401
    from transformers import BlipForQuestionAnswering, BlipProcessor

    status = availability()
    if not status.available:
        raise ModelUnavailableError(status.reason)

    d = model_dir()
    device = resolve_device()
    try:
        processor = BlipProcessor.from_pretrained(d, use_fast=True)
    except Exception:  # noqa: BLE001 - fall back to the slow processor
        processor = BlipProcessor.from_pretrained(d)
    model = BlipForQuestionAnswering.from_pretrained(d).eval().to(device)
    return model, processor, device


@lru_cache(maxsize=1)
def _cached():
    return load_cached("change-model-blip", _load)


def model_version() -> str:
    return "blip-vqa-cdvqa-external (single-image, 384px)"


def _build_pair_image(t1_image, t2_image, convention: str):
    """Present a bi-temporal pair to a single-image model under the frozen convention."""
    from PIL import Image

    if convention == "t1":
        return t1_image
    if convention == "t2":
        return t2_image
    if convention == "side_by_side":
        w, h = t1_image.size
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(t1_image, (0, 0))
        canvas.paste(t2_image, (w, 0))
        return canvas
    raise ValueError(f"Unknown pair convention: {convention!r}")


def _generate(image, question: str, max_new_tokens: int = 8) -> tuple[str, str, float | None]:
    """Run one real BLIP generation. Returns (normalized_answer, raw_text, mean_token_prob)."""
    import torch

    model, processor, device = _cached()
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
    seq = out.sequences[0]
    raw = processor.decode(seq, skip_special_tokens=True)

    # Uncalibrated generation confidence: mean softmax probability of the generated tokens.
    confidence: float | None = None
    if getattr(out, "scores", None):
        probs = []
        gen_tokens = seq[-len(out.scores):]
        for logits, token_id in zip(out.scores, gen_tokens, strict=False):
            p = torch.softmax(logits[0], dim=-1)[int(token_id)].item()
            probs.append(p)
        if probs:
            confidence = float(sum(probs) / len(probs))

    return _normalize_answer(raw), raw, confidence


def predict_single(image, question: str, max_new_tokens: int = 8) -> ChangeModelPrediction:
    """Answer a question about a single RGB image with the BLIP checkpoint."""
    _, _, device = _cached()
    answer, raw, conf = _generate(image, question, max_new_tokens)
    return ChangeModelPrediction(
        answer=answer,
        raw=raw,
        confidence=conf,
        convention="single_image",
        model_version=model_version(),
        device=device,
    )


def predict_pair(
    t1_image,
    t2_image,
    question: str,
    convention: str = FROZEN_CONVENTION,
    max_new_tokens: int = 8,
) -> ChangeModelPrediction:
    """Answer a bi-temporal question by presenting the pair under the frozen convention."""
    _, _, device = _cached()
    image = _build_pair_image(t1_image, t2_image, convention)
    answer, raw, conf = _generate(image, question, max_new_tokens)
    return ChangeModelPrediction(
        answer=answer,
        raw=raw,
        confidence=conf,
        convention=convention,
        model_version=model_version(),
        device=device,
    )
