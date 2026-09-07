"""Phase 15 — REAL training: improve CDVQA accuracy using change_model as a representation source.

Controlled experiment. The ENTIRE pipeline is identical to the Phase 9 baseline
(`scripts/train_cd_vqa.py`) — same splits, same train-only vocabularies, same GRU+fusion
head, same optimizer/epochs/seed, same val-based checkpoint selection, same frozen
Test/Test2 evaluated once. The ONLY variable is the frozen image feature source:

  --features clip       512-d CLIP-rsicd (the Phase 9 baseline representation)
  --features blip        768-d vision features from the external change_model (BLIP)
  --features clip_blip   1280-d concatenation of both  (change_model AS AN ADDED SOURCE)

This is a genuine use of the already-trained change_model as a transfer-learning /
representation source: its CDVQA-fine-tuned BLIP vision encoder provides image features to a
newly trained classifier head. Real gradient updates occur on the head.

Discipline: TRAIN fits; VAL selects; TEST/TEST2 are evaluated ONCE and never tuned on. The
change_model weights are read-only; new artifacts go to models/phase15/ only.

    python scripts/phase15/train_cdvqa_transfer.py --features clip_blip --epochs 15 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
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

CLIP_CACHE = REPO / "models" / "phase9" / "cdvqa" / "clip_features.npz"
BLIP_CACHE = REPO / "models" / "phase15" / "blip_features.npz"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_blip_features(second_dir: Path, files: list[str]) -> dict[str, np.ndarray]:
    """Cache L2-normalized BLIP (change_model) vision-encoder features for T1 and T2."""
    if BLIP_CACHE.exists():
        data = np.load(BLIP_CACHE, allow_pickle=True)
        index = {name: i for i, name in enumerate(data["files"].tolist())}
        if all(f in index for f in files):
            log(f"BLIP feature cache hit: {BLIP_CACHE} ({len(index)} images)")
            return {"files": data["files"], "f1": data["f1"], "f2": data["f2"]}

    from PIL import Image
    from transformers import BlipForQuestionAnswering, BlipProcessor

    from ml.inference.change_model_infer import model_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"extracting BLIP (change_model) vision features on {device} for {len(files)} images...")
    d = model_dir()
    try:
        proc = BlipProcessor.from_pretrained(d, use_fast=True)
    except Exception:  # noqa: BLE001
        proc = BlipProcessor.from_pretrained(d)
    model = BlipForQuestionAnswering.from_pretrained(d).eval().to(device)
    vision = model.vision_model

    f1_rows, f2_rows = [], []
    batch = 8
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(files), batch):
            chunk = files[start : start + batch]
            im1 = [Image.open(second_dir / "im1" / f).convert("RGB") for f in chunk]
            im2 = [Image.open(second_dir / "im2" / f).convert("RGB") for f in chunk]
            for imgs, rows in ((im1, f1_rows), (im2, f2_rows)):
                px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
                out = vision(pixel_values=px)
                feat = out.pooler_output  # (B, 768)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                rows.append(feat.float().cpu().numpy())
            if start % (batch * 25) == 0:
                log(f"  {start}/{len(files)} ({time.time() - started:.0f}s)")
    f1 = np.concatenate(f1_rows)
    f2 = np.concatenate(f2_rows)
    BLIP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(BLIP_CACHE, files=np.array(files), f1=f1, f2=f2)
    log(f"cached BLIP features -> {BLIP_CACHE} in {time.time() - started:.0f}s (dim {f1.shape[1]})")
    return {"files": np.array(files), "f1": f1, "f2": f2}


def _aligned(cache: dict, files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {name: i for i, name in enumerate(list(cache["files"]))}
    rows = [index[f] for f in files]
    return cache["f1"][rows], cache["f2"][rows]


def build_features(
    mode: str, files: list[str], second_dir: Path
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (f1, f2, dim) aligned to `files` for the chosen feature source."""
    parts1, parts2 = [], []
    if mode in ("clip", "clip_blip"):
        if not CLIP_CACHE.exists():
            raise FileNotFoundError(
                f"CLIP cache missing: {CLIP_CACHE} (run Phase 9 training first)"
            )
        clip = np.load(CLIP_CACHE, allow_pickle=True)
        c1, c2 = _aligned({"files": clip["files"], "f1": clip["f1"], "f2": clip["f2"]}, files)
        parts1.append(c1)
        parts2.append(c2)
    if mode in ("blip", "clip_blip"):
        blip = extract_blip_features(second_dir, files)
        b1, b2 = _aligned(blip, files)
        parts1.append(b1)
        parts2.append(b2)
    f1 = np.concatenate(parts1, axis=1)
    f2 = np.concatenate(parts2, axis=1)
    return f1.astype(np.float32), f2.astype(np.float32), f1.shape[1]


def build_tensors(samples, feat_index, answer_vocab, word_vocab, max_len):
    q = torch.tensor([encode_question(s.question, word_vocab, max_len) for s in samples])
    idx = torch.tensor([feat_index[s.img_file] for s in samples], dtype=torch.long)
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
    per_type = {t: {"acc": round(c / n, 4), "n": n, "correct": c}
                for t, (c, n) in sorted(by_type.items())}
    return acc, per_type, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["clip", "blip", "clip_blip"], default="clip_blip")
    ap.add_argument("--cdvqa-dir", default=str(REPO / "datasets" / "cdvqa" / "CDVQA-main"))
    ap.add_argument("--second-dir", default=r"D:\dataset\second")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = REPO / "models" / "phase15" / f"cdvqa_{args.features}"
    out.mkdir(parents=True, exist_ok=True)
    reports = REPO / "reports" / "phase15"
    reports.mkdir(parents=True, exist_ok=True)

    data = {s: load_split(args.cdvqa_dir, s) for s in SPLITS}
    for s in SPLITS:
        log(f"{s}: {len(data[s])} QA over {len(unique_images(data[s]))} images")
    leakage = check_leakage(data)
    log(f"image leakage (must be 0 for Train/Val/Test): {leakage}")
    assert leakage.get("Train_vs_Val", 0) == 0 and leakage.get("Train_vs_Test", 0) == 0, leakage

    answer_vocab = build_answer_vocab(data["Train"])
    answer_list = list(answer_vocab)
    word_vocab = build_word_vocab(data["Train"])
    log(f"answer classes: {len(answer_vocab)}  word vocab: {len(word_vocab)}")

    all_files = sorted({f for s in SPLITS for f in unique_images(data[s])})
    feat_index = {name: i for i, name in enumerate(all_files)}
    f1_np, f2_np, dim = build_features(args.features, all_files, Path(args.second_dir))
    log(f"feature source={args.features} dim={dim}")
    f1 = torch.tensor(f1_np, device=device)
    f2 = torch.tensor(f2_np, device=device)

    tensors = {s: build_tensors(data[s], feat_index, answer_vocab, word_vocab, args.max_len)
               for s in SPLITS}
    loaders = {s: DataLoader(TensorDataset(*tensors[s]), batch_size=args.batch_size,
                             shuffle=(s == "Train")) for s in SPLITS}

    model = CDVQAModel(len(word_vocab), len(answer_vocab), image_feat_dim=dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model params: {n_params:,} (trainable {trainable:,})")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_val, best_state, history, opt_steps = -1.0, None, [], 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for qb, ib, yb in loaders["Train"]:
            qb, ib, yb = qb.to(device), ib.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(qb, f1[ib], f2[ib])
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            opt_steps += 1
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

    test_acc, test_by_type, test_rows = evaluate(
        model, loaders["Test"], f1, f2, device, data["Test"], answer_list)
    test2_acc, test2_by_type, test2_rows = evaluate(
        model, loaders["Test2"], f1, f2, device, data["Test2"], answer_list)
    log(f"BEST val={best_val:.4f}  TEST={test_acc:.4f}  TEST2={test2_acc:.4f}  steps={opt_steps}")

    ckpt = out / "cdvqa_transfer_best.pt"
    torch.save(best_state, ckpt)
    (out / "answer_vocab.json").write_text(json.dumps(answer_vocab, indent=2))
    (out / "word_vocab.json").write_text(json.dumps(word_vocab, indent=2))
    config = {
        "experiment": f"cdvqa_{args.features}", "features": args.features, "feature_dim": dim,
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "optimizer": "Adam", "loss": "CrossEntropyLoss(ignore_index=-1)",
        "max_len": args.max_len, "device": device,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "image_encoder": {
            "clip": "flax-community/clip-rsicd-v2 (frozen)",
            "blip": "change_model BLIP vision encoder (frozen)",
            "clip_blip": "CLIP-rsicd + change_model BLIP vision (both frozen, concatenated)",
        }[args.features],
        "params_total": n_params, "params_trainable": trainable, "optimizer_steps": opt_steps,
        "train_seconds": round(train_seconds, 1),
        "answer_classes": len(answer_vocab), "word_vocab": len(word_vocab),
        "torch": torch.__version__, "cuda": torch.version.cuda, "leakage": leakage,
        "checkpoint": str(ckpt.relative_to(REPO)),
        "checkpoint_sha256": _sha256(ckpt),
        "checkpoint_bytes": ckpt.stat().st_size,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    result = {
        "experiment_id": f"cdvqa_{args.features}",
        "model": "CDVQAModel (GRU + fusion head)",
        "base_checkpoint": (
            "change_model/ (BLIP vision features)" if "blip" in args.features else None
        ),
        "dataset": "CDVQA (real) over SECOND imagery",
        "train_samples": len(data["Train"]), "validation_samples": len(data["Val"]),
        "test_samples": len(data["Test"]), "test2_samples": len(data["Test2"]),
        "epochs": args.epochs, "optimizer_steps": opt_steps,
        "best_val_accuracy": round(best_val, 4),
        "test_accuracy": round(test_acc, 4), "test2_accuracy": round(test2_acc, 4),
        "test_accuracy_by_type": test_by_type, "test2_accuracy_by_type": test2_by_type,
        "history": history, "checkpoint": str(ckpt.relative_to(REPO)),
        "sha256": config["checkpoint_sha256"], "config": config,
        "status": "trained",
    }
    (reports / f"result_{args.features}.json").write_text(json.dumps(result, indent=2))
    with (reports / f"errors_{args.features}.jsonl").open("w", encoding="utf-8") as fh:
        for r in test_rows + test2_rows:
            if not r["correct"]:
                fh.write(json.dumps(r) + "\n")
    log(f"saved -> {ckpt} and {reports / f'result_{args.features}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
