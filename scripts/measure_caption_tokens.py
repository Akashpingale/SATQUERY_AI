"""Measure how much of each BigEarthNet.txt caption survives CLIP's 77-token context.

CLIP's text encoder has a fixed 77-token context (75 content tokens plus BOS/EOS), and the
BigEarthNet.txt captions are long structured descriptions. This script tokenises the real
prepared captions with the real tokenizer so the truncation loss documented in
``docs/datasets.md`` is a measured number rather than an estimate.

    python scripts/measure_caption_tokens.py

Writes ``scripts/caption_token_report.txt``.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREPARED = REPO_ROOT / "data" / "bigearthnet" / "prepared"
REPORT = Path(__file__).resolve().parent / "caption_token_report.txt"

BASE_MODEL = "flax-community/clip-rsicd-v2"
CLIP_LIMIT = 77

LINES: list[str] = []


def emit(message: str = "") -> None:
    """Print and buffer a line (the report file is the reliable copy)."""
    print(message, flush=True)
    LINES.append(message)


def flush_report() -> None:
    with REPORT.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(LINES) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_captions(split: str) -> list[str]:
    path = PREPARED / f"{split}.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)["caption"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    from transformers import CLIPTokenizerFast

    tokenizer = CLIPTokenizerFast.from_pretrained(BASE_MODEL)

    emit("=" * 74)
    emit("CAPTION TOKEN LENGTH vs CLIP's 77-TOKEN CONTEXT")
    emit("=" * 74)
    emit(f"  tokenizer : {BASE_MODEL}")
    emit(f"  limit     : {CLIP_LIMIT} tokens (includes BOS + EOS)")
    emit()

    overall: list[int] = []
    for split in ("train", "validation", "test", "bench"):
        captions = load_captions(split)
        if not captions:
            emit(f"  {split:<11}: split file not found, skipped")
            continue

        # add_special_tokens=True counts the BOS/EOS that consume two of the 77 slots.
        lengths = [
            len(ids)
            for ids in tokenizer(
                captions, add_special_tokens=True, truncation=False, verbose=False
            )["input_ids"]
        ]
        overall.extend(lengths)

        over = sum(1 for n in lengths if n > CLIP_LIMIT)
        kept = [min(n, CLIP_LIMIT) / n for n in lengths]
        emit(f"  {split:<11}: n={len(lengths):,}")
        emit(
            f"      tokens        min={min(lengths)} median={int(statistics.median(lengths))} "
            f"mean={statistics.mean(lengths):.1f} max={max(lengths)}"
        )
        emit(f"      truncated     {over:,} / {len(lengths):,} ({100 * over / len(lengths):.1f}%)")
        emit(f"      tokens kept   mean {100 * statistics.mean(kept):.1f}% of each caption")
        emit()

    if overall:
        over = sum(1 for n in overall if n > CLIP_LIMIT)
        kept = statistics.mean(min(n, CLIP_LIMIT) / n for n in overall)
        emit("-" * 74)
        emit(f"  ALL SPLITS  : n={len(overall):,}")
        emit(
            f"      tokens        min={min(overall)} "
            f"median={int(statistics.median(overall))} "
            f"mean={statistics.mean(overall):.1f} max={max(overall)}"
        )
        emit(f"      truncated     {over:,} ({100 * over / len(overall):.1f}%)")
        emit(f"      tokens kept   mean {100 * kept:.1f}% of each caption")
        emit()
        emit("Report written to scripts/caption_token_report.txt")

    flush_report()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - report the failure, do not hide it
        emit(f"FAILED: {type(error).__name__}: {error}")
        flush_report()
        raise
