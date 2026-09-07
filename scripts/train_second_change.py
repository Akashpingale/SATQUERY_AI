"""Train a learned SECOND binary change detector (real data) and compare to Phase 6 CVA.

Adds a LEARNED detector alongside the deterministic Phase 6 CVA (which is NOT removed). Trains a
compact Siamese U-Net on the SECOND train pairs, selects on a held-out val split, and evaluates
once on a held-out internal test split. Also runs a deterministic radiometric-difference baseline
on the same held-out pairs for an honest learned-vs-deterministic comparison.

    python scripts/train_second_change.py --epochs 12 --batch-size 8 --size 256 --seed 42

TRAIN only is used for fitting; VAL for selection; the held-out split is evaluated once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from ml.datasets.second import default_root, list_pairs, load_example, make_splits  # noqa: E402
from ml.evaluation.metrics import segmentation_metrics  # noqa: E402
from ml.training.second_model import SiameseUNet  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def log(m: str) -> None:
    print(m, flush=True)


class SecondDataset(Dataset):
    def __init__(self, root, files, size):
        self.root, self.files, self.size = root, files, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        a1, a2, mask = load_example(self.root, self.files[i], self.size)
        x = np.concatenate([a1, a2], axis=0)  # (6,H,W)
        return torch.from_numpy(x), torch.from_numpy(mask)


def dice_bce_loss(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum((2, 3))
    union = probs.sum((2, 3)) + target.sum((2, 3))
    dice = 1 - ((2 * inter + 1) / (union + 1)).mean()
    return bce + dice


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = tn = 0
    for x, y in loader:
        x = x.to(device)
        pred = (torch.sigmoid(model(x)).cpu().numpy()[:, 0] > 0.5)
        gt = y.numpy() > 0.5
        seg = segmentation_metrics(pred, gt)
        tp += seg.true_positive
        fp += seg.false_positive
        fn += seg.false_negative
        tn += seg.true_negative
    union = tp + fp + fn
    iou = tp / union if union else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"iou": round(iou, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


@torch.no_grad()
def deterministic_baseline(root, files, size):
    """Radiometric-difference reference (Otsu on |T2-T1| magnitude) on the same held-out set.

    A deterministic reference in the spirit of the Phase 6 CVA (radiometric change), computed on
    the SECOND PNGs. Labelled a reference — not the exact production cva-robust configuration."""
    tp = fp = fn = 0
    for f in files:
        a1, a2, mask = load_example(root, f, size)
        mag = np.sqrt(((a2 - a1) ** 2).sum(0))
        vals = mag.reshape(-1)
        # Otsu threshold
        hist, edges = np.histogram(vals, bins=64)
        centers = (edges[:-1] + edges[1:]) / 2
        w = hist.cumsum() / hist.sum()
        mu = (hist * centers).cumsum() / hist.sum()
        mu_t = mu[-1]
        denom = w * (1 - w)
        sigma_b = np.where(denom > 0, (mu_t * w - mu) ** 2 / np.where(denom > 0, denom, 1), 0)
        thr = centers[int(sigma_b.argmax())]
        pred = mag > thr
        gt = mask > 0.5
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
    union = tp + fp + fn
    iou = tp / union if union else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"iou": round(iou, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(default_root()))
    ap.add_argument("--output-dir", default=str(REPO / "models" / "phase9" / "second"))
    ap.add_argument("--reports-dir", default=str(REPO / "reports" / "benchmarks" / "second"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="cap train pairs (smoke); 0 = all")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    files = list_pairs(args.data_dir)
    splits = make_splits(files, seed=args.seed)
    log(f"pairs={len(files)} splits={splits.as_counts()}")
    train_files = splits.train[: args.limit] if args.limit else splits.train

    train_ds = SecondDataset(args.data_dir, train_files, args.size)
    val_ds = SecondDataset(args.data_dir, splits.val, args.size)
    test_ds = SecondDataset(args.data_dir, splits.test, args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    model = SiameseUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    log(f"model params: {n_params:,} device={device}")

    best_iou, best_state, history = -1.0, None, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.unsqueeze(1).to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = dice_bce_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss) * x.size(0)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": round(total / len(train_ds), 4), **val})
        log(f"epoch {epoch}: loss={history[-1]['train_loss']} val_iou={val['iou']} "
            f"val_f1={val['f1']}")
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.time() - started

    assert best_state is not None
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    log(f"BEST val IoU={best_iou}  TEST {test}")
    log("computing deterministic baseline on held-out test...")
    baseline = deterministic_baseline(args.data_dir, splits.test, args.size)
    log(f"deterministic baseline TEST {baseline}")

    torch.save(best_state, out / "second_best.pt")
    config = {
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "size": args.size, "lr": args.lr, "device": device, "params": n_params,
        "train_seconds": round(train_seconds, 1), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "split_seed": args.seed, "splits": splits.as_counts(),
        "task": "binary change (derived from SECOND semantic labels: changed = not no-change)",
        "train_pairs_used": len(train_files),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    metrics = {
        "benchmark": "SECOND binary change (internal held-out split)",
        "dataset_status": "additional public dataset (NOT SIH-provided)",
        "splits": splits.as_counts(),
        "learned_model": {"best_val_iou": round(best_iou, 4), "test": test},
        "deterministic_baseline_reference": baseline,
        "history": history,
        "config": config,
        "notes": (
            "Binary change derived from semantic labels (documented). Held-out split is an "
            "INTERNAL split of the provided SECOND train set, not the official SECOND test set. "
            "Baseline is a deterministic radiometric-difference reference (Otsu), in the spirit "
            "of Phase 6 CVA, computed on the same held-out pairs."
        ),
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "config.json").write_text(json.dumps(config, indent=2))
    # split sample ids for reproducibility / leakage audit
    (reports / "splits.json").write_text(json.dumps(
        {"train": splits.train, "val": splits.val, "test": splits.test}, indent=2))
    log(f"saved to {out} and {reports}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
