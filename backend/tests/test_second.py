"""Phase 9 SECOND change-detection tests.

Data/checkpoint-dependent tests skip when the SECOND dataset or the trained checkpoint are
absent (large/optional), so CI without them stays green. Unit tests need neither.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ml.datasets import second as ds
from ml.training.second_model import SiameseUNet

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("SECOND_DIR", r"D:\dataset\second"))
CKPT = REPO / "models" / "phase9" / "second" / "second_best.pt"

have_data = (DATA / "im1").is_dir()
have_ckpt = CKPT.is_file()
needs_data = pytest.mark.skipif(not have_data, reason="SECOND dataset not provisioned")
needs_ckpt = pytest.mark.skipif(not have_ckpt, reason="SECOND checkpoint not present")


# --- unit tests (no data) ----------------------------------------------------------


def test_binary_change_mask_from_rgb():
    lab = np.full((4, 4, 3), 255, dtype="uint8")  # all no-change (white)
    lab[1, 1] = (128, 0, 0)  # a changed (buildings) pixel
    mask = ds.binary_change_mask(lab)
    assert mask.sum() == 1
    assert mask[1, 1]
    assert not mask[0, 0]


def test_make_splits_deterministic_and_disjoint():
    files = [f"{i:05d}.png" for i in range(100)]
    a = ds.make_splits(files, seed=42)
    b = ds.make_splits(files, seed=42)
    assert a.train == b.train and a.test == b.test  # deterministic
    s_train, s_val, s_test = set(a.train), set(a.val), set(a.test)
    assert not (s_train & s_val) and not (s_train & s_test) and not (s_val & s_test)
    assert len(a.train) + len(a.val) + len(a.test) == 100


def test_model_forward_shape():
    model = SiameseUNet()
    x = torch.randn(2, 6, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


# --- data / checkpoint tests -------------------------------------------------------


@needs_data
def test_load_example_shapes_and_mask_range():
    files = ds.list_pairs(DATA)
    a1, a2, mask = ds.load_example(DATA, files[0], size=128)
    assert a1.shape == (3, 128, 128) and a2.shape == (3, 128, 128)
    assert mask.shape == (128, 128)
    assert set(np.unique(mask).tolist()) <= {0.0, 1.0}


@needs_ckpt
def test_second_checkpoint_reloads():
    model = SiameseUNet()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    out = model(torch.randn(1, 6, 64, 64))
    assert out.shape == (1, 1, 64, 64)


def test_second_specialist_registered_but_not_default_route():
    from app.agent.planner import TASK_TOOLCHAIN
    from app.agent.specialists import SPECIALISTS
    from app.registry.models import MODEL_REGISTRY
    from app.registry.tasks import TaskType

    assert "second_change_learned" in SPECIALISTS
    assert "second_change_learned" in MODEL_REGISTRY
    # Deterministic CVA remains the default temporal-change route.
    assert "second_change_learned" not in TASK_TOOLCHAIN[TaskType.TEMPORAL_CHANGE]
    assert TASK_TOOLCHAIN[TaskType.TEMPORAL_CHANGE] == ["registration_validator", "temporal_change"]
