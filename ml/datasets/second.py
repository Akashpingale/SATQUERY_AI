"""SECOND change-detection dataset loader (additional public dataset, NOT SIH-provided).

SECOND provides per-date semantic land-cover labels (label1/label2) as RGB-coded PNGs, where
unchanged pixels are white (255,255,255) in both dates and changed pixels carry their
land-cover class. This loader:

  - Derives a **binary change mask** = pixels that are NOT no-change (white) in label1. This is
    an explicit, documented reduction of the semantic labels to a binary changed/unchanged task
    so a learned detector can be compared directly against the deterministic Phase 6 CVA.
    (Full from-to semantic change is preserved in the raw labels and left as future work.)
  - Creates deterministic train/val/test splits from the provided TRAIN set (the archive has no
    official val/test), with a fixed seed and recorded sample ids. The held-out portion is an
    INTERNAL test split, not the official SECOND test set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NO_CHANGE_RGB = (255, 255, 255)


def default_root() -> Path:
    return Path(os.environ.get("SECOND_DIR", r"D:\dataset\second"))


def list_pairs(root: str | Path) -> list[str]:
    """Sorted list of image file names present in im1/ (and by construction im2/label1/label2)."""
    return sorted(os.listdir(Path(root) / "im1"))


@dataclass(frozen=True)
class Splits:
    train: list[str]
    val: list[str]
    test: list[str]

    def as_counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def make_splits(files: list[str], seed: int = 42, ratios=(0.7, 0.15, 0.15)) -> Splits:
    """Deterministic shuffle-and-partition. Disjoint by construction."""
    rng = np.random.default_rng(seed)
    order = list(files)
    rng.shuffle(order)
    n = len(order)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return Splits(
        train=sorted(order[:n_train]),
        val=sorted(order[n_train : n_train + n_val]),
        test=sorted(order[n_train + n_val :]),
    )


def binary_change_mask(label_rgb: np.ndarray) -> np.ndarray:
    """Boolean changed-pixel mask from an RGB semantic label (changed = not no-change/white)."""
    if label_rgb.ndim != 3 or label_rgb.shape[-1] < 3:
        raise ValueError("expected an HxWx3 RGB label array")
    r, g, b = label_rgb[..., 0], label_rgb[..., 1], label_rgb[..., 2]
    is_white = (r == NO_CHANGE_RGB[0]) & (g == NO_CHANGE_RGB[1]) & (b == NO_CHANGE_RGB[2])
    return ~is_white


def load_example(root: str | Path, file: str, size: int = 256):
    """Load one example: (im1, im2) float tensors in [0,1] and a binary change mask.

    Images are bilinearly resized; the label is nearest-resized so class colours are preserved.
    Returns numpy arrays: im1 (3,H,W), im2 (3,H,W), mask (H,W) float32.
    """
    from PIL import Image

    root = Path(root)
    im1 = Image.open(root / "im1" / file).convert("RGB").resize((size, size), Image.BILINEAR)
    im2 = Image.open(root / "im2" / file).convert("RGB").resize((size, size), Image.BILINEAR)
    lab = Image.open(root / "label1" / file).convert("RGB").resize((size, size), Image.NEAREST)
    a1 = np.asarray(im1, dtype="float32").transpose(2, 0, 1) / 255.0
    a2 = np.asarray(im2, dtype="float32").transpose(2, 0, 1) / 255.0
    mask = binary_change_mask(np.asarray(lab)).astype("float32")
    return a1, a2, mask
