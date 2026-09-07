"""Phase 11 — LEVIR-CD public binary change-detection evaluation.

Real, reproducible cross-dataset evaluation on the public LEVIR-CD test split
(ericyu/LEVIRCD_Cropped_256, 256x256 crops, binary changed/unchanged masks). This is a
CROSS-DATASET generalization test: the learned detector was trained ONLY on SECOND (Phase 9),
never on LEVIR-CD, and LEVIR-CD labels *building* change specifically while the SECOND model
learned *any* land-cover change — so degradation is expected and reported honestly.

Evaluated, like-for-like (binary segmentation IoU/F1/precision/recall):
  1. Learned SECOND Siamese U-Net (Phase 9 checkpoint, frozen) — generalization test.
  2. Deterministic radiometric-difference reference (Otsu on |T2-T1|) — the same deterministic
     baseline used in Phase 9, as a cross-dataset reference.

No training, no tuning, no test-set fitting. Reads the parquet directly with pyarrow (avoids the
repo-local `datasets/` folder that shadows the HF library).

    python scripts/phase11/eval_levir_cd.py --parquet D:\dataset\levir_cd\test.parquet --limit 0
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from ml.evaluation.metrics import mean_and_ci, segmentation_metrics  # noqa: E402
from ml.training.second_model import SiameseUNet  # noqa: E402

OUT = REPO / "reports" / "phase11"
SECOND_CKPT = REPO / "models" / "phase9" / "second" / "second_best.pt"


def _decode(cell) -> Image.Image:
    data = cell["bytes"] if isinstance(cell, dict) else cell
    return Image.open(io.BytesIO(data))


def _to_chw(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype="float32").transpose(2, 0, 1) / 255.0


def otsu_threshold(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=64)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return float(values.max())
    w = hist.cumsum() / total
    mu = (hist * centers).cumsum() / total
    mu_t = mu[-1]
    denom = w * (1 - w)
    sigma_b = np.where(denom > 0, (mu_t * w - mu) ** 2 / np.where(denom > 0, denom, 1), 0)
    return float(centers[int(sigma_b.argmax())])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=r"D:\dataset\levir_cd\test.parquet")
    ap.add_argument("--limit", type=int, default=0, help="0 = full test split")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--name", default="LEVIR-CD (public test split)")
    ap.add_argument("--source", default="huggingface: ericyu/LEVIRCD_Cropped_256 (256x256 crops)")
    ap.add_argument("--out-file", default="levir_cd_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = pq.read_table(args.parquet)
    cols = table.column_names
    # Column names may be imageA/imageB/label; fall back to positional if flattened.
    names = {"a": "imageA", "b": "imageB", "m": "label"}
    if not all(col in cols for col in names.values()):
        names = {"a": cols[0], "b": cols[1], "m": cols[2]}
    n_rows = table.num_rows
    n = n_rows if args.limit in (0, None) else min(args.limit, n_rows)
    print(f"LEVIR-CD test rows={n_rows}  evaluating n={n}  columns={cols}  device={device}",
          flush=True)

    a_col = table[names["a"]].to_pylist()
    b_col = table[names["b"]].to_pylist()
    m_col = table[names["m"]].to_pylist()

    model = None
    if SECOND_CKPT.is_file():
        model = SiameseUNet().to(device)
        model.load_state_dict(torch.load(SECOND_CKPT, map_location=device))
        model.eval()
        print("loaded SECOND learned checkpoint", flush=True)
    else:
        print("SECOND checkpoint absent; only the deterministic reference will run", flush=True)

    # Accumulators
    learned = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "ious": []}
    otsu = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "ious": []}
    pos_pixels = 0
    total_pixels = 0

    started = time.time()
    batch_imgs: list[np.ndarray] = []
    batch_gt: list[np.ndarray] = []
    batch_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    def flush_batch():
        if not batch_imgs:
            return
        if model is not None:
            x = torch.from_numpy(np.stack(batch_imgs)).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
            for pred, gt in zip(probs > 0.5, batch_gt, strict=True):
                seg = segmentation_metrics(pred, gt)
                learned["tp"] += seg.true_positive
                learned["fp"] += seg.false_positive
                learned["fn"] += seg.false_negative
                learned["tn"] += seg.true_negative
                if gt.any():
                    learned["ious"].append(seg.iou)
        for (a1, a2), gt in zip(batch_pairs, batch_gt, strict=True):
            mag = np.sqrt(((a2 - a1) ** 2).sum(0))
            pred = mag > otsu_threshold(mag.reshape(-1))
            seg = segmentation_metrics(pred, gt)
            otsu["tp"] += seg.true_positive
            otsu["fp"] += seg.false_positive
            otsu["fn"] += seg.false_negative
            otsu["tn"] += seg.true_negative
            if gt.any():
                otsu["ious"].append(seg.iou)
        batch_imgs.clear()
        batch_gt.clear()
        batch_pairs.clear()

    for i in range(n):
        a1 = _to_chw(_decode(a_col[i]))
        a2 = _to_chw(_decode(b_col[i]))
        mask_img = _decode(m_col[i]).convert("L")
        gt = np.asarray(mask_img) > 127
        pos_pixels += int(gt.sum())
        total_pixels += int(gt.size)
        batch_imgs.append(np.concatenate([a1, a2], axis=0))
        batch_pairs.append((a1, a2))
        batch_gt.append(gt)
        if len(batch_imgs) >= args.batch_size:
            flush_batch()
        if i % 512 == 0 and i:
            print(f"  {i}/{n} ({time.time() - started:.0f}s)", flush=True)
    flush_batch()

    def finalize(acc: dict) -> dict:
        tp, fp, fn, tn = acc["tp"], acc["fp"], acc["fn"], acc["tn"]
        union = tp + fp + fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {
            "micro_iou": round(tp / union, 4) if union else 0.0,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "pixel_accuracy": round((tp + tn) / (tp + fp + fn + tn), 4),
            "mean_iou_changed_images": mean_and_ci(acc["ious"]),
            "pixels": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }

    result = {
        "benchmark": args.name,
        "source": args.source,
        "evaluation_type": (
            "FULL TEST EVALUATION" if args.limit in (0, None) else "SUBSET EVALUATION"
        ),
        "test_rows_total": n_rows,
        "evaluated": n,
        "changed_pixel_fraction": round(pos_pixels / total_pixels, 5) if total_pixels else 0.0,
        "device": device,
        "seconds": round(time.time() - started, 1),
        "cross_dataset_note": (
            "The learned model was trained ONLY on SECOND (Phase 9) and never on this "
            "benchmark. Label semantics and sensor/resolution differ from SECOND, so this is a "
            "cross-dataset generalization test; large degradation vs the in-domain SECOND result "
            "(IoU 0.4219) is expected and is reported honestly."
        ),
        "learned_second_model": finalize(learned) if model is not None else "checkpoint absent",
        "deterministic_otsu_reference": finalize(otsu),
        "reference_note": (
            "Otsu radiometric-difference baseline (same method as the Phase 9 SECOND baseline); "
            "not the georeferenced production cva-robust config."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out_file).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cross_dataset_note",)},
                     indent=2))
    print(f"\nwrote {OUT / args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
