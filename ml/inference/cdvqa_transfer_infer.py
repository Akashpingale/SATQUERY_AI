"""Inference for the Phase 15 improved CDVQA classifier (clip_blip transfer model).

This is the newly TRAINED model (`models/phase15/cdvqa_clip_blip/`). It classifies a CDVQA
question over the closed 19-answer vocabulary using image features from TWO frozen encoders
concatenated per image:

    [ CLIP-rsicd (512) | change_model BLIP vision (768) ]  = 1280-d, for T1 and T2

i.e. it uses the already-trained external ``change_model`` as a complementary representation
source on top of the Phase 9 CLIP-rsicd features. On the frozen CDVQA benchmark it improves
over the Phase 9 baseline (Test 0.7156 vs 0.7114, Test2 0.6668 vs 0.6632; val 0.7200 vs 0.7166).

Weights are loaded read-only. Degrades gracefully via ``availability()`` if any component is
missing.
"""

from __future__ import annotations

import json
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

CLIP_MODEL = "flax-community/clip-rsicd-v2"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    configured = os.environ.get("CDVQA_TRANSFER_DIR", "models/phase15/cdvqa_clip_blip")
    path = Path(configured)
    return path if path.is_absolute() else _repo_root() / path


def availability() -> Availability:
    if not torch_available():
        return Availability(False, "PyTorch is not installed.")
    if not transformers_available():
        return Availability(False, "Transformers is not installed.")
    d = model_dir()
    needed = ["cdvqa_transfer_best.pt", "answer_vocab.json", "word_vocab.json"]
    missing = [f for f in needed if not (d / f).is_file()]
    if missing:
        return Availability(
            False, f"cdvqa_transfer checkpoint incomplete (missing {', '.join(missing)})."
        )
    from .change_model_infer import availability as blip_availability

    blip = blip_availability()
    if not blip.available:
        return Availability(False, f"change_model (BLIP feature source) unavailable: {blip.reason}")
    return Availability(True)


@dataclass
class CDVQATransferPrediction:
    answer: str
    confidence: float
    ranked: list[tuple[str, float]]
    model_version: str
    device: str


def _load():
    import torch
    from transformers import BlipForQuestionAnswering, BlipProcessor, CLIPModel, CLIPProcessor

    from ml.training.cdvqa_model import CDVQAModel

    from .change_model_infer import model_dir as blip_dir

    status = availability()
    if not status.available:
        raise ModelUnavailableError(status.reason)

    d = model_dir()
    answer_vocab = json.loads((d / "answer_vocab.json").read_text())
    word_vocab = json.loads((d / "word_vocab.json").read_text())
    answer_list = list(answer_vocab)
    device = resolve_device()

    head = CDVQAModel(len(word_vocab), len(answer_vocab), image_feat_dim=1280)
    state = torch.load(d / "cdvqa_transfer_best.pt", map_location=device)
    head.load_state_dict(state)
    head.eval().to(device)

    clip = CLIPModel.from_pretrained(CLIP_MODEL).eval().to(device)
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)

    bdir = blip_dir()
    try:
        blip_proc = BlipProcessor.from_pretrained(bdir, use_fast=True)
    except Exception:  # noqa: BLE001
        blip_proc = BlipProcessor.from_pretrained(bdir)
    blip = BlipForQuestionAnswering.from_pretrained(bdir).eval().to(device)
    blip_vision = blip.vision_model

    return head, clip, clip_proc, blip_vision, blip_proc, word_vocab, answer_list, device


@lru_cache(maxsize=1)
def _cached():
    return load_cached("cdvqa-transfer-clip-blip", _load)


def _features(image, clip, clip_proc, blip_vision, blip_proc, device):
    """Return the 1280-d [CLIP | BLIP] normalized feature vector for one PIL image."""
    import torch

    with torch.no_grad():
        cpx = clip_proc(images=image, return_tensors="pt")["pixel_values"].to(device)
        cfeat = clip.get_image_features(pixel_values=cpx)
        cfeat = cfeat / cfeat.norm(dim=-1, keepdim=True)

        bpx = blip_proc(images=image, return_tensors="pt")["pixel_values"].to(device)
        bfeat = blip_vision(pixel_values=bpx).pooler_output
        bfeat = bfeat / bfeat.norm(dim=-1, keepdim=True)

        return torch.cat([cfeat, bfeat], dim=-1)


def predict(t1_image, t2_image, question: str) -> CDVQATransferPrediction:
    """Answer a change-VQA question about a T1/T2 pair with the improved clip_blip model.

    ``t1_image`` / ``t2_image`` are PIL RGB images.
    """
    import torch

    from ml.datasets.cdvqa import encode_question

    head, clip, clip_proc, blip_vision, blip_proc, word_vocab, answer_list, device = _cached()
    f1 = _features(t1_image, clip, clip_proc, blip_vision, blip_proc, device)
    f2 = _features(t2_image, clip, clip_proc, blip_vision, blip_proc, device)
    q = torch.tensor([encode_question(question, word_vocab)], device=device)

    with torch.no_grad():
        logits = head(q, f1, f2)
        probs = logits.softmax(-1)[0]
    order = probs.argsort(descending=True).tolist()
    ranked = [(answer_list[i], float(probs[i])) for i in order[:5]]
    best = order[0]
    return CDVQATransferPrediction(
        answer=answer_list[best],
        confidence=float(probs[best]),
        ranked=ranked,
        model_version="cdvqa-clip-rsicd+blip-transfer (1280-d, 19-class)",
        device=device,
    )
