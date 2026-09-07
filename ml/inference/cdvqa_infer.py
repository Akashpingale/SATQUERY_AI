"""Inference for the Phase 9 trained CDVQA model.

Loads the trained checkpoint + vocabularies and answers a change-VQA question about a T1/T2
image pair using the SAME frozen CLIP image features and question encoding as training. Returns
the predicted answer (closed vocabulary) with a softmax confidence.

If the checkpoint or its dependencies are absent, `availability()` reports it and the specialist
degrades gracefully rather than fabricating an answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .base import Availability, ModelUnavailableError, load_cached, resolve_device

CLIP_MODEL = "flax-community/clip-rsicd-v2"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    import os

    configured = os.environ.get("CDVQA_MODEL_DIR", "models/phase9/cdvqa")
    path = Path(configured)
    return path if path.is_absolute() else _repo_root() / path


def availability() -> Availability:
    d = model_dir()
    needed = ["cdvqa_best.pt", "answer_vocab.json", "word_vocab.json"]
    missing = [f for f in needed if not (d / f).is_file()]
    if missing:
        return Availability(False, f"CDVQA checkpoint incomplete (missing {', '.join(missing)}).")
    return Availability(True)


@dataclass
class CDVQAPrediction:
    answer: str
    confidence: float
    ranked: list[tuple[str, float]]
    model_version: str


def _load():
    import torch

    from ml.training.cdvqa_model import CDVQAModel

    status = availability()
    if not status.available:
        raise ModelUnavailableError(status.reason)

    d = model_dir()
    answer_vocab = json.loads((d / "answer_vocab.json").read_text())
    word_vocab = json.loads((d / "word_vocab.json").read_text())
    answer_list = list(answer_vocab)
    device = resolve_device()

    model = CDVQAModel(len(word_vocab), len(answer_vocab))
    state = torch.load(d / "cdvqa_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)

    from transformers import CLIPModel, CLIPProcessor

    clip = CLIPModel.from_pretrained(CLIP_MODEL).eval().to(device)
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return model, clip, proc, word_vocab, answer_list, device


@lru_cache(maxsize=1)
def _cached():
    return load_cached("cdvqa-trained", _load)


def _clip_feature(clip, proc, image, device):
    import torch

    with torch.no_grad():
        px = proc(images=image, return_tensors="pt")["pixel_values"].to(device)
        feat = clip.get_image_features(pixel_values=px)
        return feat / feat.norm(dim=-1, keepdim=True)


def predict(t1_path: str, t2_path: str, question: str) -> CDVQAPrediction:
    """Answer a change-VQA question about a T1/T2 pair with the trained model."""
    import torch
    from PIL import Image

    from ml.datasets.cdvqa import encode_question

    model, clip, proc, word_vocab, answer_list, device = _cached()
    f1 = _clip_feature(clip, proc, Image.open(t1_path).convert("RGB"), device)
    f2 = _clip_feature(clip, proc, Image.open(t2_path).convert("RGB"), device)
    q = torch.tensor([encode_question(question, word_vocab)], device=device)

    with torch.no_grad():
        logits = model(q, f1, f2)
        probs = logits.softmax(-1)[0]
    order = probs.argsort(descending=True).tolist()
    ranked = [(answer_list[i], float(probs[i])) for i in order[:5]]
    best = order[0]
    return CDVQAPrediction(
        answer=answer_list[best],
        confidence=float(probs[best]),
        ranked=ranked,
        model_version=f"cdvqa-clip-rsicd-frozen+head ({len(answer_list)}-class)",
    )
