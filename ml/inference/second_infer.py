"""Inference for the Phase 9 learned SECOND binary change detector.

Loads the trained Siamese U-Net checkpoint and predicts a binary change mask for a co-registered
T1/T2 pair. Returns the changed-pixel fraction and a mean-probability confidence proxy
(uncalibrated). Degrades gracefully via `availability()` when the checkpoint is absent.

This is a LEARNED detector offered ALONGSIDE the deterministic Phase 6 CVA, not a replacement,
and it detects BINARY change (derived from SECOND semantic labels), not from-to semantic class.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .base import Availability, ModelUnavailableError, load_cached, resolve_device


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    import os

    configured = os.environ.get("SECOND_MODEL_DIR", "models/phase9/second")
    path = Path(configured)
    return path if path.is_absolute() else _repo_root() / path


def availability() -> Availability:
    if not (model_dir() / "second_best.pt").is_file():
        return Availability(False, "SECOND checkpoint (second_best.pt) not found.")
    return Availability(True)


@dataclass
class SecondPrediction:
    changed_fraction: float
    mean_confidence: float
    size: int
    model_version: str


def _load():
    import torch

    from ml.training.second_model import SiameseUNet

    status = availability()
    if not status.available:
        raise ModelUnavailableError(status.reason)
    device = resolve_device()
    model = SiameseUNet()
    model.load_state_dict(torch.load(model_dir() / "second_best.pt", map_location=device))
    model.eval().to(device)
    return model, device


@lru_cache(maxsize=1)
def _cached():
    return load_cached("second-change-learned", _load)


def predict(
    t1_path: str, t2_path: str, size: int = 256, threshold: float = 0.5
) -> SecondPrediction:
    """Predict a binary change mask for a T1/T2 pair with the learned SECOND detector."""
    import numpy as np
    import torch
    from PIL import Image

    model, device = _cached()
    im1 = Image.open(t1_path).convert("RGB").resize((size, size), Image.BILINEAR)
    im2 = Image.open(t2_path).convert("RGB").resize((size, size), Image.BILINEAR)
    a1 = np.asarray(im1, dtype="float32").transpose(2, 0, 1) / 255.0
    a2 = np.asarray(im2, dtype="float32").transpose(2, 0, 1) / 255.0
    x = torch.from_numpy(np.concatenate([a1, a2], axis=0)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    mask = probs > threshold
    changed = float(mask.mean())
    mean_conf = float(probs[mask].mean()) if mask.any() else float(1.0 - probs.mean())
    return SecondPrediction(
        changed_fraction=round(changed, 5),
        mean_confidence=round(mean_conf, 4),
        size=size,
        model_version="second-siamese-unet-binary (256px)",
    )
