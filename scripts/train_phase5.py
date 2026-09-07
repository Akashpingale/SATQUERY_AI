"""Run the real Phase 5 LoRA adaptation of CLIP on BigEarthNet.txt.

    python scripts/train_phase5.py [--epochs N] [--batch-size N] [--rank N]

This is the full adaptation run, distinct from ``phase5_smoke_train.py`` (which only proves
the pipeline executes). Outputs:

  models/phase5/adapter/                 the learned LoRA adapter + processor
  reports/phase5/training_config.json    exact configuration used
  reports/phase5/training_metrics.json   measured losses, timing, VRAM
  reports/phase5/environment.json        versions, GPU, git commit, seed
  scripts/phase5_training_report.txt     human-readable log

Batch size is probed downward on CUDA OOM so the run adapts to the available VRAM rather
than failing outright; the batch size actually used is recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREPARED = REPO_ROOT / "data" / "bigearthnet" / "prepared"
LMDB_DIR = REPO_ROOT / "data" / "bigearthnet" / "imagery" / "BENv2_lithuania_summer.lmdb"
MANIFEST = REPO_ROOT / "data" / "manifests" / "bigearthnet_txt_manifest.json"
ADAPTER_OUT = REPO_ROOT / "models" / "phase5" / "adapter"
REPORTS = REPO_ROOT / "reports" / "phase5"
REPORT = Path(__file__).resolve().parent / "phase5_training_report.txt"


def emit(message: str = "") -> None:
    print(message, flush=True)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.stdout.strip() or "unknown (not committed)"
    except Exception:  # noqa: BLE001
        return "unknown"


def environment_record(config, batch_size: int) -> dict:
    import peft
    import torch
    import transformers

    properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "gpu": properties.name if properties else None,
        "gpu_vram_bytes": properties.total_memory if properties else None,
        "gpu_capability": f"{properties.major}.{properties.minor}" if properties else None,
        "git_commit": git_commit(),
        "seed": config.seed,
        "batch_size_used": batch_size,
        "dataset_manifest": str(MANIFEST.relative_to(REPO_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 LoRA adaptation of CLIP.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()

    REPORT.unlink(missing_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    import torch

    from ml.training import LoraTrainingConfig, train_lora_clip

    if not torch.cuda.is_available():
        emit("ABORT: CUDA is not available. Phase 5 requires GPU adaptation.")
        return 1

    if not (PREPARED / "train.jsonl").is_file():
        emit(f"ABORT: {PREPARED / 'train.jsonl'} missing. Run prepare_bigearthnet_txt.py first.")
        return 1

    emit("=" * 74)
    emit("PHASE 5 REAL TRAINING - LoRA adaptation of CLIP on BigEarthNet.txt")
    emit("=" * 74)
    emit(f"  GPU   : {torch.cuda.get_device_name(0)}")
    emit(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    emit(f"  torch : {torch.__version__} (CUDA {torch.version.cuda})")
    emit("")

    # Probe downward on OOM rather than failing: the run should fit whatever VRAM exists.
    batch_size = args.batch_size
    metrics = None
    adapter_path = None

    while batch_size >= 4:
        config = LoraTrainingConfig(
            output_dir=str(ADAPTER_OUT),
            epochs=args.epochs,
            batch_size=batch_size,
            rank=args.rank,
            alpha=args.alpha,
            learning_rate=args.learning_rate,
            seed=args.seed,
            max_train_samples=args.max_train_samples,
        )
        emit(f"attempting training with batch_size={batch_size}")
        try:
            metrics, adapter_path = train_lora_clip(
                config, PREPARED, LMDB_DIR, progress=lambda message: emit(f"  {message}")
            )
            break
        except torch.cuda.OutOfMemoryError:
            emit(f"  CUDA OOM at batch_size={batch_size}; halving and retrying.")
            torch.cuda.empty_cache()
            batch_size //= 2
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            emit(f"  CUDA OOM at batch_size={batch_size}; halving and retrying.")
            torch.cuda.empty_cache()
            batch_size //= 2

    if metrics is None or adapter_path is None:
        emit("ABORT: training did not complete at any batch size.")
        return 1

    emit("")
    emit("=" * 74)
    emit("TRAINING COMPLETE (measured)")
    emit("=" * 74)
    emit(f"  batch size used     : {batch_size}")
    emit(
        f"  trainable params    : {metrics.trainable_parameters:,} of "
        f"{metrics.total_parameters:,}"
    )
    emit(f"  final train loss    : {metrics.final_train_loss:.4f}")
    emit(f"  final val loss      : {metrics.final_val_loss:.4f}")
    emit(f"  best val loss       : {metrics.best_val_loss:.4f}")
    emit(f"  duration            : {metrics.train_seconds / 60:.1f} min")
    emit(f"  peak VRAM allocated : {metrics.peak_vram_bytes / 1024 ** 2:.1f} MB")
    emit(f"  adapter             : {adapter_path}")
    emit("")
    emit("  per-epoch:")
    for entry in metrics.epochs:
        emit(
            f"    epoch {entry['epoch']}: train={entry['train_loss']:.4f} "
            f"val={entry['val_loss']:.4f}"
        )

    config = LoraTrainingConfig(
        output_dir=str(ADAPTER_OUT),
        epochs=args.epochs,
        batch_size=batch_size,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    (REPORTS / "training_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (REPORTS / "training_metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
    )
    (REPORTS / "environment.json").write_text(
        json.dumps(environment_record(config, batch_size), indent=2), encoding="utf-8"
    )
    emit("")
    emit(f"  wrote {REPORTS.relative_to(REPO_ROOT)}/training_config.json")
    emit(f"  wrote {REPORTS.relative_to(REPO_ROOT)}/training_metrics.json")
    emit(f"  wrote {REPORTS.relative_to(REPO_ROOT)}/environment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
