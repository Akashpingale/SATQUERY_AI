"""Evaluate the external `change_model/` BLIP checkpoint on the REAL CDVQA data.

The checkpoint is a single-image BLIP VQA model, but its answer vocabulary (e.g. `0_to_10`)
shows CDVQA fine-tuning. CDVQA samples are bi-temporal (T1 + T2), so the pair must be presented
to a single-image model somehow. The convention used by the model's author is NOT encoded in the
checkpoint, so it is determined EMPIRICALLY on the CDVQA **validation** split only, then frozen
and evaluated once on the held-out **Test** and **Test2** splits.

    python scripts/eval_change_model_cdvqa.py --val-samples 800 --test-samples 3000

Split discipline: Val is used for the input-convention choice; Test/Test2 are never used for any
tuning. Subsampling is seeded and the sample size + Wilson CI are always reported.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_DIR = REPO / "change_model"
CDVQA_DIR = REPO / "datasets" / "cdvqa" / "CDVQA-main"
SECOND_DIR = Path(os.environ.get("SECOND_DIR", r"D:\dataset\second"))
OUT = REPO / "reports" / "change_model"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from ml.datasets.cdvqa import load_split  # noqa: E402
from ml.evaluation.metrics import accuracy_ci  # noqa: E402


def normalize_answer(text: str) -> str:
    """Normalise a generated answer for comparison against the CDVQA reference.

    The BERT word-piece decoder renders CDVQA's underscore tokens with spaces
    (``"0 _ to _ 10"`` for ``"0_to_10"``), so spaces around underscores are collapsed. Only
    formatting is normalised - no synonym mapping, so accuracy is not inflated.
    """
    t = text.strip().lower()
    t = re.sub(r"\s*_\s*", "_", t)
    t = " ".join(t.split())
    return t.rstrip(".!?")


def build_image(convention: str, t1, t2):
    """Present a bi-temporal pair to a single-image model under a given convention."""
    from PIL import Image

    if convention == "t1":
        return t1
    if convention == "t2":
        return t2
    if convention == "side_by_side":
        w, h = t1.size
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(t1, (0, 0))
        canvas.paste(t2, (w, 0))
        return canvas
    if convention == "stacked":
        w, h = t1.size
        canvas = Image.new("RGB", (w, h * 2))
        canvas.paste(t1, (0, 0))
        canvas.paste(t2, (0, h))
        return canvas
    raise ValueError(convention)


def run(model, processor, device, samples, convention: str, batch_size: int) -> list[dict]:
    """Generate answers for `samples` under one input convention."""
    import torch
    from PIL import Image

    cache: dict[str, tuple] = {}

    def pair(file: str):
        if file not in cache:
            cache[file] = (
                Image.open(SECOND_DIR / "im1" / file).convert("RGB"),
                Image.open(SECOND_DIR / "im2" / file).convert("RGB"),
            )
        return cache[file]

    rows: list[dict] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        images = []
        questions = []
        for s in chunk:
            t1, t2 = pair(s.img_file)
            images.append(build_image(convention, t1, t2))
            questions.append(s.question)
        inputs = processor(images=images, text=questions, return_tensors="pt", padding=True).to(
            device
        )
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8)
        for s, seq in zip(chunk, out, strict=True):
            raw = processor.decode(seq, skip_special_tokens=True)
            pred = normalize_answer(raw)
            rows.append(
                {
                    "qid": s.question_id,
                    "type": s.qtype,
                    "truth": s.answer,
                    "raw": raw,
                    "pred": pred,
                    "correct": pred == s.answer,
                }
            )
        if start % (batch_size * 20) == 0:
            done = min(start + batch_size, len(samples))
            acc = sum(r["correct"] for r in rows) / max(1, len(rows))
            print(f"    {done}/{len(samples)} running_acc={acc:.4f}", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    by_type = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by_type[r["type"]][0] += int(r["correct"])
        by_type[r["type"]][1] += 1
    correct = sum(r["correct"] for r in rows)
    return {
        "samples": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "accuracy_ci_wilson": accuracy_ci(correct, len(rows)),
        "accuracy_by_type": {t: round(c / n, 4) for t, (c, n) in sorted(by_type.items())},
        "samples_by_type": {t: n for t, (_c, n) in sorted(by_type.items())},
    }


def main() -> int:
    import torch
    from transformers import BlipForQuestionAnswering, BlipProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--val-samples", type=int, default=800)
    ap.add_argument("--test-samples", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = BlipProcessor.from_pretrained(MODEL_DIR, use_fast=True)
    except Exception:  # noqa: BLE001
        processor = BlipProcessor.from_pretrained(MODEL_DIR)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_DIR).eval().to(device)
    print(f"model on {device}")

    def subsample(split: str, n: int):
        data = load_split(CDVQA_DIR, split)
        if n and n < len(data):
            return rng.sample(data, n), len(data)
        return data, len(data)

    report: dict = {
        "model_dir": str(MODEL_DIR),
        "architecture": "BlipForQuestionAnswering (single-image, 384px, 3ch)",
        "device": device,
        "seed": args.seed,
        "note": (
            "Single-image model evaluated on bi-temporal CDVQA. The pair-presentation "
            "convention was selected on the VALIDATION split only; Test/Test2 were evaluated "
            "once with the frozen choice. Subsampled with a fixed seed; Wilson CIs reported."
        ),
    }

    # --- 1. select the input convention on VALIDATION only ---------------------
    val, val_total = subsample("Val", args.val_samples)
    print(f"\n=== convention selection on VAL ({len(val)} of {val_total}) ===")
    conventions = ["t1", "t2", "side_by_side"]
    val_results = {}
    for conv in conventions:
        t0 = time.time()
        rows = run(model, processor, device, val, conv, args.batch_size)
        summary = summarize(rows)
        summary["seconds"] = round(time.time() - t0, 1)
        val_results[conv] = summary
        print(f"  {conv:14} val_acc={summary['accuracy']:.4f}  ({summary['seconds']}s)")
    best = max(conventions, key=lambda c: val_results[c]["accuracy"])
    print(f"  -> selected convention: {best} (val acc {val_results[best]['accuracy']:.4f})")
    report["val_convention_selection"] = val_results
    report["selected_convention"] = best
    report["val_subsample"] = {"used": len(val), "total": val_total}

    # --- 2. held-out evaluation with the frozen convention --------------------
    for split in ["Test", "Test2"]:
        data, total = subsample(split, args.test_samples)
        print(f"\n=== {split} ({len(data)} of {total}) with convention '{best}' ===")
        t0 = time.time()
        rows = run(model, processor, device, data, best, args.batch_size)
        summary = summarize(rows)
        summary["seconds"] = round(time.time() - t0, 1)
        summary["subsample"] = {"used": len(data), "total": total}
        report[split.lower()] = summary
        print(f"  {split} accuracy={summary['accuracy']:.4f} "
              f"CI={summary['accuracy_ci_wilson']['ci_low']}-"
              f"{summary['accuracy_ci_wilson']['ci_high']} ({summary['seconds']}s)")
        print(f"  by type: {summary['accuracy_by_type']}")
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / f"cdvqa_{split.lower()}_predictions.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    # Comparison against the Phase 9 trained classifier (measured previously, full splits).
    report["phase9_classifier_reference"] = {
        "test_accuracy": 0.7114,
        "test2_accuracy": 0.6632,
        "note": "Measured on the FULL Test/Test2 splits (reports/benchmarks/cdvqa/metrics.json).",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cdvqa_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'cdvqa_evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
