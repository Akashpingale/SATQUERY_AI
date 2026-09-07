"""Phase 9 CDVQA tests.

Dataset-dependent tests skip gracefully when the CDVQA JSON or the trained checkpoint are
not present (they are large/optional and gitignored), so CI without the data stays green.
No multi-GB downloads and no GPU are required by these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ml.datasets import cdvqa as ds
from ml.training.cdvqa_model import CDVQAModel

REPO = Path(__file__).resolve().parents[2]
CDVQA_DIR = REPO / "datasets" / "cdvqa" / "CDVQA-main"
MODEL_DIR = REPO / "models" / "phase9" / "cdvqa"

have_data = (CDVQA_DIR / "Val_questions.json").is_file()
have_ckpt = (MODEL_DIR / "cdvqa_best.pt").is_file()
needs_data = pytest.mark.skipif(not have_data, reason="CDVQA dataset not provisioned")
needs_ckpt = pytest.mark.skipif(not have_ckpt, reason="CDVQA checkpoint not present")


# --- model unit tests (no data needed) --------------------------------------------


def test_model_forward_shapes():
    model = CDVQAModel(word_vocab_size=50, num_answers=19)
    q = torch.randint(0, 50, (4, 20))
    f1 = torch.randn(4, 512)
    f2 = torch.randn(4, 512)
    out = model(q, f1, f2)
    assert out.shape == (4, 19)


def test_encode_question_pads_and_truncates():
    vocab = {ds.PAD: 0, ds.UNK: 1, "did": 2, "anything": 3, "change": 4}
    ids = ds.encode_question("Did anything change?", vocab, max_len=6)
    assert len(ids) == 6
    assert ids[:3] == [2, 3, 4]
    assert ids[-1] == 0  # padded
    # Unknown words map to <unk>.
    assert ds.encode_question("zzz", vocab, max_len=3)[0] == 1


# --- dataset tests (need the provisioned JSON) ------------------------------------


@needs_data
def test_load_split_joins_records():
    val = ds.load_split(CDVQA_DIR, "Val")
    assert len(val) > 1000
    s = val[0]
    assert s.img_file.endswith(".png")
    assert s.question and s.answer and s.qtype
    assert s.answer == s.answer.lower()


@needs_data
def test_no_image_leakage_between_train_val_test():
    splits = {name: ds.load_split(CDVQA_DIR, name) for name in ["Train", "Val", "Test"]}
    leakage = ds.check_leakage(splits)
    assert leakage["Train_vs_Val"] == 0
    assert leakage["Train_vs_Test"] == 0
    assert leakage["Val_vs_Test"] == 0


@needs_data
def test_vocab_built_from_train_only_is_closed():
    train = ds.load_split(CDVQA_DIR, "Train")
    answers = ds.build_answer_vocab(train)
    words = ds.build_word_vocab(train)
    assert len(answers) >= 2
    assert "yes" in answers and "no" in answers
    assert words[ds.PAD] == 0 and words[ds.UNK] == 1


# --- checkpoint reload + inference wiring ------------------------------------------


@needs_ckpt
def test_checkpoint_reloads_into_model():
    import json

    answer_vocab = json.loads((MODEL_DIR / "answer_vocab.json").read_text())
    word_vocab = json.loads((MODEL_DIR / "word_vocab.json").read_text())
    model = CDVQAModel(len(word_vocab), len(answer_vocab))
    state = torch.load(MODEL_DIR / "cdvqa_best.pt", map_location="cpu")
    model.load_state_dict(state)  # raises if the architecture/shapes mismatch
    model.eval()
    q = torch.zeros(1, 20, dtype=torch.long)
    out = model(q, torch.randn(1, 512), torch.randn(1, 512))
    assert out.shape == (1, len(answer_vocab))


@needs_ckpt
def test_cdvqa_inference_availability_reports_present():
    from ml.inference import cdvqa_infer

    assert cdvqa_infer.availability().available is True


def test_cdvqa_trained_specialist_is_registered():
    from app.agent.specialists import SPECIALISTS

    assert "cdvqa_trained" in SPECIALISTS


def test_cdvqa_trained_in_registry_but_not_default_route():
    from app.agent.planner import TASK_TOOLCHAIN
    from app.registry.models import MODEL_REGISTRY
    from app.registry.tasks import TaskType

    assert "cdvqa_trained" in MODEL_REGISTRY
    # The deterministic detector remains the default change-VQA route (generalization/robustness).
    assert "cdvqa_trained" not in TASK_TOOLCHAIN[TaskType.CHANGE_VQA]
    assert TASK_TOOLCHAIN[TaskType.CHANGE_VQA] == ["registration_validator", "change_vqa"]
