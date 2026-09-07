"""Train a real CDVQA constrained-classifier on the actual CDVQA + SECOND imagery.

Pipeline (all real data — no synthetic substitution):
  1. Cache frozen CLIP image features for every unique SECOND image (T1 + T2).
  2. Build closed answer vocab + question word vocab from the TRAIN split only.
  3. Train the CDVQA classifier; select the best checkpoint on the VAL split.
  4. Evaluate the held-out TEST and TEST2 splits once, with the frozen best checkpoint.

    python scripts/train_cd_vqa.py --epochs 15 --batch-size 256 --seed 42

TRAIN only is used for fitting; VAL for selection; TEST/TEST2 are never used for tuning.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(REPO / "models" / "hf-cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from ml.datasets.cdvqa import (  # noqa: E402
    SPLITS,
    build_answer_vocab,
    build_word_vocab,
    check_leakage,
    encode_question,
    load_split,
    unique_images,
)
from ml.training.cdvqa_model import CDVQAModel  # noqa: E402

CLIP_MODEL = "flax-community/clip-rsicd-v2"


try:
    sys.stdout.reconfigure(encoding="utf-8")  # avoid cp1252 crashes on Windows consoles
except Exception:  # noqa: BLE001
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def cache_image_features(second_dir: Path, files: list[str], cache_path: Path) -> dict[str, int]:
    """Compute (or load) frozen CLIP image features for T1 and T2 of each file.

    Returns a mapping file_name -> row index into the saved feature matrices.
    """
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        index = {name: i for i, name in enumerate(data["files"].tolist())}
        if all(f in index for f in files):
            log(f"feature cache hit: {cache_path} ({len(index)} images)")
            return index
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"extracting CLIP features on {device} for {len(files)} images (T1+T2)...")
    model = CLIPModel.from_pretrained(CLIP_MODEL).eval().to(device)
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)

    f1_rows, f2_rows = [], []
    batch = 32
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(files), batch):
            chunk = files[start : start + batch]
            im1 = [Image.open(second_dir / "im1" / f).convert("RGB") for f in chunk]
            im2 = [Image.open(second_dir / "im2" / f).convert("RGB") for f in chunk]
            for imgs, rows in ((im1, f1_rows), (im2, f2_rows)):
                px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
                feats = model.get_image_features(pixel_values=px)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                rows.append(feats.float().cpu().numpy())
            if start % (batch * 20) == 0:
                log(f"  {start}/{len(files)} ({time.time() - started:.0f}s)")
    f1 = np.concatenate(f1_rows)
    f2 = np.concatenate(f2_rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, files=np.array(files), f1=f1, f2=f2)
    log(f"cached features -> {cache_path} in {time.time() - started:.0f}s")
    return {name: i for i, name in enumerate(files)}


def build_tensors(samples, feat_index, answer_vocab, word_vocab, max_len):
    q = torch.tensor([encode_question(s.question, word_vocab, max_len) for s in samples])
    idx = torch.tensor([feat_index[s.img_file] for s in samples], dtype=torch.long)
    # Unknown answers (not in the closed train vocab) get -1 and are treated as never-correct.
    y = torch.tensor([answer_vocab.get(s.answer, -1) for s in samples], dtype=torch.long)
    return q, idx, y


def evaluate(model, loader, f1, f2, device, samples, answer_list):
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for qb, ib, _ in loader:
            qb = qb.to(device)
            logits = model(qb, f1[ib.to(device)], f2[ib.to(device)])
            preds.extend(logits.argmax(-1).cpu().tolist())
    correct = 0
    by_type = collections.defaultdict(lambda: [0, 0])
    rows = []
    for s, p in zip(samples, preds, strict=True):
        pred_ans = answer_list[p]
        ok = pred_ans == s.answer
        correct += int(ok)
        by_type[s.qtype][0] += int(ok)
        by_type[s.qtype][1] += 1
        rows.append({"qid": s.question_id, "type": s.qtype, "truth": s.answer,
                     "pred": pred_ans, "correct": ok})
    acc = correct / len(samples) if samples else 0.0
    per_type = {t: round(c / n, 4) for t, (c, n) in sorted(by_type.items())}
    return acc, per_type, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdvqa-dir", default=str(REPO / "datasets" / "cdvqa" / "CDVQA-main"))
    ap.add_argument("--second-dir", default=r"D:\dataset\second")
    ap.add_argument("--output-dir", default=str(REPO / "models" / "phase9" / "cdvqa"))
    ap.add_argument("--reports-dir", default=str(REPO / "reports" / "benchmarks" / "cdvqa"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    # --- load real splits -------------------------------------------------------
    data = {s: load_split(args.cdvqa_dir, s) for s in SPLITS}
    for s in SPLITS:
        log(f"{s}: {len(data[s])} QA over {len(unique_images(data[s]))} images")
    leakage = check_leakage(data)
    log(f"image leakage (must be 0 for Train/Val/Test): {leakage}")

    answer_vocab = build_answer_vocab(data["Train"])
    answer_list = list(answer_vocab)
    word_vocab = build_word_vocab(data["Train"])
    log(f"answer classes: {len(answer_vocab)}  word vocab: {len(word_vocab)}")

    # --- cache frozen image features for every referenced image -----------------
    all_files = sorted({f for s in SPLITS for f in unique_images(data[s])})
    feat_index = cache_image_features(
        Path(args.second_dir), all_files, out / "clip_features.npz"
    )
    cache = np.load(out / "clip_features.npz", allow_pickle=True)
    f1 = torch.tensor(cache["f1"], device=device)
    f2 = torch.tensor(cache["f2"], device=device)

    tensors = {s: build_tensors(data[s], feat_index, answer_vocab, word_vocab, args.max_len)
               for s in SPLITS}
    loaders = {
        s: DataLoader(TensorDataset(*tensors[s]), batch_size=args.batch_size,
                      shuffle=(s == "Train"))
        for s in SPLITS
    }

    model = CDVQAModel(len(word_vocab), len(answer_vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model params: {n_params:,} (trainable {trainable:,})")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_val, best_state, history = -1.0, None, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for qb, ib, yb in loaders["Train"]:
            qb, ib, yb = qb.to(device), ib.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(qb, f1[ib], f2[ib])
            loss = loss_fn(logits, yb.clamp(min=0)) if (yb >= 0).all() else loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss) * qb.size(0)
        val_acc, _, _ = evaluate(model, loaders["Val"], f1, f2, device, data["Val"], answer_list)
        history.append({"epoch": epoch, "train_loss": round(total / len(data["Train"]), 4),
                        "val_acc": round(val_acc, 4)})
        log(f"epoch {epoch}: train_loss={history[-1]['train_loss']} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.time() - started

    assert best_state is not None
    model.load_state_dict(best_state)

    # --- held-out evaluation (once) --------------------------------------------
    test_acc, test_by_type, test_rows = evaluate(
        model, loaders["Test"], f1, f2, device, data["Test"], answer_list)
    test2_acc, test2_by_type, test2_rows = evaluate(
        model, loaders["Test2"], f1, f2, device, data["Test2"], answer_list)
    log(f"BEST val_acc={best_val:.4f}  TEST acc={test_acc:.4f}  TEST2 acc={test2_acc:.4f}")

    # majority-per-type baseline (train priors) for honest comparison
    train_major = {}
    grouped = collections.defaultdict(collections.Counter)
    for s in data["Train"]:
        grouped[s.qtype][s.answer] += 1
    for t, c in grouped.items():
        train_major[t] = c.most_common(1)[0][0]

    def baseline_acc(samples):
        return round(sum(1 for s in samples if train_major.get(s.qtype) == s.answer)
                     / len(samples), 4)

    # --- save everything --------------------------------------------------------
    torch.save(best_state, out / "cdvqa_best.pt")
    (out / "answer_vocab.json").write_text(json.dumps(answer_vocab, indent=2))
    (out / "word_vocab.json").write_text(json.dumps(word_vocab, indent=2))
    config = {
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "max_len": args.max_len, "device": device,
        "image_encoder": CLIP_MODEL + " (frozen)", "feature_dim": 512,
        "params_total": n_params, "params_trainable": trainable,
        "train_seconds": round(train_seconds, 1),
        "answer_classes": len(answer_vocab), "word_vocab": len(word_vocab),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "leakage": leakage,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    metrics = {
        "benchmark": "CDVQA (real held-out)",
        "sih_status": "SIH26167 multitemporal change-VQA benchmark",
        "samples": {s: len(data[s]) for s in SPLITS},
        "best_val_accuracy": round(best_val, 4),
        "test_accuracy": round(test_acc, 4),
        "test2_accuracy": round(test2_acc, 4),
        "test_accuracy_by_type": test_by_type,
        "test2_accuracy_by_type": test2_by_type,
        "baseline_majority_per_type": {
            "test": baseline_acc(data["Test"]), "test2": baseline_acc(data["Test2"])},
        "history": history,
        "config": config,
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (reports / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in test_rows + test2_rows:
            fh.write(json.dumps(r) + "\n")
    with (reports / "errors.jsonl").open("w", encoding="utf-8") as fh:
        for r in test_rows + test2_rows:
            if not r["correct"]:
                fh.write(json.dumps(r) + "\n")
    (reports / "config.json").write_text(json.dumps(config, indent=2))
    log(f"saved checkpoint + metrics to {out} and {reports}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
