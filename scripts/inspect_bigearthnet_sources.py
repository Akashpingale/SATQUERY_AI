"""Inspect BigEarthNet.txt and BigEarthNet v2.0 (reBEN) sources: real sizes, no assumptions.

    python scripts/inspect_bigearthnet_sources.py

Queries the official Hugging Face and Zenodo APIs for actual file sizes so the download
plan can be based on measured numbers. Writes ``scripts/bigearthnet_sources_report.txt``.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

REPORT = Path(__file__).resolve().parent / "bigearthnet_sources_report.txt"

HF_TXT = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
HF_LMDB_SUBSET = "hackelle/BigEarthNetV2-Lithuania-Summer-LMDB"
ZENODO_REBEN = "10891137"


def emit(message: str = "") -> None:
    print(message, flush=True)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def get_json(url: str) -> dict | list | None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "satquery-ai/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        emit(f"  ! request failed: {url} -> {exc}")
        return None


def head_size(url: str) -> int | None:
    """Content-Length for a URL, following redirects to the CDN."""
    try:
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "satquery-ai/0.1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        emit(f"  ! HEAD failed: {url} -> {exc}")
        return None


def mb(value: float | None) -> str:
    return "unknown" if value is None else f"{value / 1024 / 1024:.1f} MB"


def gb(value: float | None) -> str:
    return "unknown" if value is None else f"{value / 1024 / 1024 / 1024:.2f} GB"


def main() -> int:
    REPORT.unlink(missing_ok=True)

    emit("=" * 78)
    emit("1. BigEarthNet.txt annotations (official, Hugging Face)")
    emit("=" * 78)
    info = get_json(f"https://huggingface.co/api/datasets/{HF_TXT}")
    if info:
        card = info.get("cardData") or {}
        emit(f"  repo      : {HF_TXT}")
        emit(f"  gated     : {info.get('gated')}")
        emit(f"  license   : {card.get('license')}")
        emit(f"  downloads : {info.get('downloads')}")
        for name in ("BigEarthNet.txt.parquet", "ben_txt_datamodule.py"):
            url = f"https://huggingface.co/datasets/{HF_TXT}/resolve/main/{name}"
            emit(f"  {name}: {mb(head_size(url))}")

    emit()
    emit("=" * 78)
    emit("2. Pre-converted LMDB image subset (UNOFFICIAL mirror, Hugging Face)")
    emit("=" * 78)
    info = get_json(f"https://huggingface.co/api/datasets/{HF_LMDB_SUBSET}")
    if info:
        card = info.get("cardData") or {}
        emit(f"  repo      : {HF_LMDB_SUBSET}")
        emit(f"  gated     : {info.get('gated')}")
        emit(f"  license   : {card.get('license')}")
        emit(f"  size_cat  : {card.get('size_categories')}")
        total = 0
        for sibling in info.get("siblings", []):
            name = sibling["rfilename"]
            url = f"https://huggingface.co/datasets/{HF_LMDB_SUBSET}/resolve/main/{name}"
            size = head_size(url)
            if size:
                total += size
            emit(f"  {name}: {mb(size)}")
        emit(f"  TOTAL: {gb(total)}")

    emit()
    emit("=" * 78)
    emit("3. reBEN / BigEarthNet v2.0 original imagery (official, Zenodo)")
    emit("=" * 78)
    record = get_json(f"https://zenodo.org/api/records/{ZENODO_REBEN}")
    if record:
        emit(f"  title   : {record.get('title')}")
        emit(f"  doi     : {record.get('doi')}")
        licence = ((record.get("metadata") or {}).get("license") or {})
        emit(f"  license : {licence.get('id') or licence}")
        files = record.get("files") or []
        total = 0
        for entry in files:
            size = entry.get("size")
            if size:
                total += size
            emit(f"  {entry.get('key')}: {gb(size)}")
        emit(f"  TOTAL OFFICIAL IMAGERY: {gb(total)}")

    emit()
    emit("=" * 78)
    emit("4. Local disk")
    emit("=" * 78)
    for drive in ("C:\\", "D:\\"):
        try:
            usage = shutil.disk_usage(drive)
            emit(f"  {drive} free: {gb(usage.free)} of {gb(usage.total)}")
        except OSError as exc:
            emit(f"  {drive} -> {exc}")

    emit()
    emit("Report written to scripts/bigearthnet_sources_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
