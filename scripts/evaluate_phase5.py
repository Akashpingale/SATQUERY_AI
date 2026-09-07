"""Evaluate the Phase 4 baseline against the Phase 5 adapted model.

    python scripts/evaluate_phase5.py [--split test] [--limit N]

Both models are evaluated with the identical dataset split, preprocessing, and batch size,
so the difference reported is attributable to the adapter rather than to the harness.

Metric: bidirectional image-text retrieval (Recall@1/5/10, median and mean rank) on a
held-out split that was never used for training or validation. Retrieval is the appropriate
measurement for a contrastively trained dual encoder.

Outputs:
  reports/phase5/evaluation_metrics.json
  scripts/phase5_evaluation_report.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREPARED = REPO_ROOT / "data" / "bigearthnet" / "prepared"
LMDB_DIR = REPO_ROOT / "data" / "bigearthnet" / "imagery" / "BENv2_lithuania_summer.lmdb"
ADAPTER = REPO_ROOT / "models" / "phase5" / "adapter"
REPORTS = REPO_ROOT / "reports" / "phase5"
REPORT = Path(__file__).resolve().parent / "phase5_evaluation_report.txt"

BASE_MODEL = "flax-community/clip-rsicd-v2"


def emit(message: str = "") -> None:
    print(message, flush=True)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def show(label: str, metrics) -> None:
    emit(f"  {label}")
    emit(
        f"    image->text  R@1={metrics.i2t_recall_at_1:.4f}  R@5={metrics.i2t_recall_at_5:.4f}  "
        f"R@10={metrics.i2t_recall_at_10:.4f}  median rank={metrics.i2t_median_rank:.1f}"
    )
    emit(
        f"    text->image  R@1={metrics.t2i_recall_at_1:.4f}  R@5={metrics.t2i_recall_at_5:.4f}  "
        f"R@10={metrics.t2i_recall_at_10:.4f}  median rank={metrics.t2i_median_rank:.1f}"
    )
    emit(f"    mean recall  {metrics.mean_recall:.4f}   (encode {metrics.encode_seconds:.1f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline vs adapted retrieval evaluation.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    REPORT.unlink(missing_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import CLIPModel, CLIPProcessor

    from ml.evaluation import evaluate_retrieval, random_baseline_recall
    from ml.training import load_adapted_clip, load_split

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emit("=" * 74)
    emit("PHASE 5 EVALUATION - baseline vs adapted (image-text retrieval)")
    emit("=" * 74)
    emit(f"  device : {device}")
    emit(f"  split  : {args.split} (held out; never used for training or validation)")

    if not ADAPTER.is_dir() or not any(ADAPTER.glob("adapter_model*")):
        emit("")
        emit(f"NOT EVALUATED - no adapter found at {ADAPTER}.")
        emit("Run scripts/train_phase5.py first.")
        return 1

    rows = load_split(PREPARED, args.split, args.limit)
    emit(f"  samples: {len(rows):,}")
    chance = random_baseline_recall(len(rows))
    emit(
        f"  chance : R@1={chance['recall_at_1']:.5f} R@5={chance['recall_at_5']:.5f} "
        f"R@10={chance['recall_at_10']:.5f} median rank={chance['median_rank']:.0f}"
    )
    emit("")

    # --- baseline -----------------------------------------------------------------
    emit("Evaluating BASELINE (Phase 4 model, no adapter)...")
    baseline_model = CLIPModel.from_pretrained(BASE_MODEL).to(device).eval()
    baseline_processor = CLIPProcessor.from_pretrained(BASE_MODEL)
    baseline = evaluate_retrieval(
        baseline_model,
        baseline_processor,
        rows,
        LMDB_DIR,
        model_label=f"baseline:{BASE_MODEL}",
        device=device,
        batch_size=args.batch_size,
    )
    show("BASELINE", baseline)
    del baseline_model
    if device == "cuda":
        torch.cuda.empty_cache()
    emit("")

    # --- adapted ------------------------------------------------------------------
    emit("Evaluating ADAPTED (Phase 5 LoRA adapter)...")
    adapted_model, adapted_processor = load_adapted_clip(ADAPTER, device=device)
    adapted = evaluate_retrieval(
        adapted_model,
        adapted_processor,
        rows,
        LMDB_DIR,
        model_label="adapted:phase5-lora",
        device=device,
        batch_size=args.batch_size,
    )
    show("ADAPTED", adapted)
    emit("")

    # --- measured difference ------------------------------------------------------
    emit("=" * 74)
    emit("MEASURED DIFFERENCE (adapted - baseline)")
    emit("=" * 74)
    comparisons = [
        ("image->text R@1", baseline.i2t_recall_at_1, adapted.i2t_recall_at_1),
        ("image->text R@5", baseline.i2t_recall_at_5, adapted.i2t_recall_at_5),
        ("image->text R@10", baseline.i2t_recall_at_10, adapted.i2t_recall_at_10),
        ("text->image R@1", baseline.t2i_recall_at_1, adapted.t2i_recall_at_1),
        ("text->image R@5", baseline.t2i_recall_at_5, adapted.t2i_recall_at_5),
        ("text->image R@10", baseline.t2i_recall_at_10, adapted.t2i_recall_at_10),
        ("mean recall", baseline.mean_recall, adapted.mean_recall),
    ]
    for name, before, after in comparisons:
        delta = after - before
        arrow = "+" if delta > 0 else ("-" if delta < 0 else "=")
        emit(f"  {name:20s} {before:.4f} -> {after:.4f}   {arrow}{abs(delta):.4f}")

    emit(
        f"  {'i2t median rank':20s} {baseline.i2t_median_rank:.1f} -> "
        f"{adapted.i2t_median_rank:.1f}"
    )
    emit(
        f"  {'t2i median rank':20s} {baseline.t2i_median_rank:.1f} -> "
        f"{adapted.t2i_median_rank:.1f}"
    )

    payload = {
        "task": "image-text retrieval",
        "dataset": "BigEarthNet.txt (Lithuania/Summer captioning subset)",
        "split": args.split,
        "samples": len(rows),
        "note": (
            "Held-out split from the dataset's official `split` column; not used for "
            "training or validation. Both models evaluated with identical preprocessing."
        ),
        "chance_level": chance,
        "baseline": baseline.to_dict(),
        "adapted": adapted.to_dict(),
        "difference": {
            name: round(after - before, 6) for name, before, after in comparisons
        },
    }
    (REPORTS / "evaluation_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    emit("")
    emit(f"wrote {(REPORTS / 'evaluation_metrics.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
