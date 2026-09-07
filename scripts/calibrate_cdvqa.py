"""Confidence calibration for the trained CDVQA model (real held-out data).

Temperature scaling is fit on the VAL split (minimising NLL) and then evaluated, frozen, on
the held-out TEST split. Reports ECE/Brier before and after. Never fits on test labels.

    python scripts/calibrate_cdvqa.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ml.datasets.cdvqa import (  # noqa: E402
    build_answer_vocab,
    build_word_vocab,
    encode_question,
    load_split,
)
from ml.evaluation.calibration import reliability  # noqa: E402
from ml.training.cdvqa_model import CDVQAModel  # noqa: E402

CDVQA = REPO / "datasets" / "cdvqa" / "CDVQA-main"
OUT = REPO / "models" / "phase9" / "cdvqa"
REPORTS = REPO / "reports" / "benchmarks" / "cdvqa"


def logits_for(model, samples, feat_index, f1, f2, word_vocab, device):
    q = torch.tensor([encode_question(s.question, word_vocab) for s in samples], device=device)
    idx = torch.tensor([feat_index[s.img_file] for s in samples], device=device)
    out = []
    with torch.no_grad():
        for start in range(0, len(samples), 1024):
            sl = slice(start, start + 1024)
            out.append(model(q[sl], f1[idx[sl]], f2[idx[sl]]).cpu())
    return torch.cat(out)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = load_split(CDVQA, "Train")
    val = load_split(CDVQA, "Val")
    test = load_split(CDVQA, "Test")
    answer_vocab = build_answer_vocab(train)
    word_vocab = build_word_vocab(train)

    cache = np.load(OUT / "clip_features.npz", allow_pickle=True)
    feat_index = {n: i for i, n in enumerate(cache["files"].tolist())}
    f1 = torch.tensor(cache["f1"], device=device)
    f2 = torch.tensor(cache["f2"], device=device)

    model = CDVQAModel(len(word_vocab), len(answer_vocab)).to(device)
    model.load_state_dict(torch.load(OUT / "cdvqa_best.pt", map_location=device))
    model.eval()

    val_logits = logits_for(model, val, feat_index, f1, f2, word_vocab, device)
    test_logits = logits_for(model, test, feat_index, f1, f2, word_vocab, device)
    val_y = torch.tensor([answer_vocab.get(s.answer, -1) for s in val])
    test_y = torch.tensor([answer_vocab.get(s.answer, -1) for s in test])

    # Fit temperature on VAL by minimising NLL over answerable samples.
    mask = val_y >= 0
    logit_v, y_v = val_logits[mask], val_y[mask]
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=60)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logit_v / T.clamp(min=1e-2), y_v)
        loss.backward()
        return loss

    opt.step(closure)
    temperature = float(T.detach().clamp(min=1e-2))

    def calib(logits, y, temp):
        probs = (logits / temp).softmax(-1)
        conf, pred = probs.max(-1)
        correct = [(int(p) == int(t)) if t >= 0 else False for p, t in zip(pred, y, strict=True)]
        return conf.tolist(), correct

    conf_u, corr_u = calib(test_logits, test_y, 1.0)
    conf_c, corr_c = calib(test_logits, test_y, temperature)
    before = reliability(conf_u, corr_u, n_bins=10, note="TEST, uncalibrated (T=1)")
    after = reliability(
        conf_c, corr_c, n_bins=10, note=f"TEST, temperature-scaled (T={temperature:.3f})"
    )

    payload = {
        "benchmark": "CDVQA confidence calibration (real held-out)",
        "method": "temperature scaling fit on VAL, evaluated frozen on TEST",
        "temperature": round(temperature, 4),
        "val_samples": int(mask.sum()),
        "test_samples": len(test),
        "uncalibrated": {"ece": before.ece, "mce": before.mce, "brier": before.brier,
                          "overall_confidence": before.overall_confidence,
                          "overall_accuracy": before.overall_accuracy},
        "calibrated": {"ece": after.ece, "mce": after.mce, "brier": after.brier,
                        "overall_confidence": after.overall_confidence,
                        "overall_accuracy": after.overall_accuracy},
        "reliability_bins_calibrated": after.as_dict()["bins"],
    }
    (REPORTS / "calibration.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
