"""Prepare the Phase 5 image-caption dataset from BigEarthNet.txt.

    python scripts/prepare_bigearthnet_txt.py

Filters the official annotation file to the country/season covered by the downloaded
imagery, keeps ``captioning`` rows, verifies that every retained sample has a decodable
image, writes deterministic split files using the dataset's **official** split column, and
records a reproducible manifest with SHA256 hashes.

Outputs:
  data/bigearthnet/prepared/{train,validation,test,bench}.jsonl
  data/manifests/bigearthnet_txt_manifest.json
  scripts/bigearthnet_prepare_report.txt
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data" / "bigearthnet"
ANNOTATIONS = DATA / "annotations" / "BigEarthNet.txt.parquet"
IMAGERY_DIR = DATA / "imagery"
LMDB_DIR = IMAGERY_DIR / "BENv2_lithuania_summer.lmdb"
IMAGERY_METADATA = IMAGERY_DIR / "metadata_lithuania_summer.parquet"
PREPARED = DATA / "prepared"
MANIFEST = REPO_ROOT / "data" / "manifests" / "bigearthnet_txt_manifest.json"
REPORT = Path(__file__).resolve().parent / "bigearthnet_prepare_report.txt"

COUNTRY = "Lithuania"
SEASON = "Summer"
SEED = 42
PREPROCESSING_VERSION = "phase5-rgb-v1"


def emit(message: str = "") -> None:
    print(message, flush=True)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    REPORT.unlink(missing_ok=True)

    if not ANNOTATIONS.is_file():
        emit(f"MISSING {ANNOTATIONS} - run scripts/download_bigearthnet.py")
        return 1

    from ml.datasets import LmdbImageReader, prepare_caption_dataset, write_manifest

    emit("=" * 74)
    emit("PREPARING BigEarthNet.txt IMAGE-CAPTION DATASET")
    emit("=" * 74)
    emit(f"  annotations : {ANNOTATIONS.name}")
    emit(f"  imagery     : {LMDB_DIR.name}")
    emit(f"  filters     : country={COUNTRY} season={SEASON} type=captioning")
    emit("  split source: official `split` column")
    emit("")

    samples, report = prepare_caption_dataset(
        ANNOTATIONS, LMDB_DIR, country=COUNTRY, season=SEASON, seed=SEED
    )

    emit("COUNTS (measured)")
    emit(f"  annotations scanned      : {report.annotations_scanned:,}")
    emit(f"  matched country+season   : {report.matched_country_season:,}")
    emit(f"  captioning rows          : {report.captioning_rows:,}")
    emit(f"  retained                 : {report.retained:,}")
    emit(f"  excluded (no image)      : {report.excluded_missing_image:,}")
    emit(f"  excluded (undecodable)   : {report.excluded_unreadable_image:,}")
    emit(f"  excluded (empty caption) : {report.excluded_empty_caption:,}")
    emit(f"  excluded (duplicate)     : {report.excluded_duplicate:,}")
    emit(f"  per split                : {report.per_split}")
    emit(f"  unique patches           : {len({s.patch_id for s in samples}):,}")
    if report.excluded_examples:
        emit("  exclusion examples:")
        for item in report.excluded_examples[:5]:
            emit(f"    {item}")
    emit("")

    if not samples:
        emit("ABORT: no samples retained.")
        return 1

    emit("SAMPLE CAPTIONS (real text from the dataset)")
    for sample in samples[:5]:
        emit(f"  [{sample.split}] {sample.patch_id}")
        emit(f"      {sample.caption}")
    lengths = [len(s.caption.split()) for s in samples]
    emit(
        f"  caption words: min={min(lengths)} mean={sum(lengths) / len(lengths):.1f} "
        f"max={max(lengths)}"
    )
    emit("")

    # Confirm the RGB conversion produces the expected 10 m resolution shape.
    reader = LmdbImageReader(LMDB_DIR)
    rgb = reader.read_rgb(samples[0].patch_id)
    if rgb is None:
        emit("ABORT: first sample image could not be decoded.")
        return 1
    emit(f"RGB image check: shape={rgb.shape} dtype={rgb.dtype} range={rgb.min()}..{rgb.max()}")
    reader.close()
    emit("")

    PREPARED.mkdir(parents=True, exist_ok=True)
    for split in sorted(report.per_split):
        rows = [s for s in samples if s.split == split]
        if not rows:
            continue
        path = PREPARED / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for sample in rows:
                handle.write(
                    json.dumps(
                        {
                            "patch_id": sample.patch_id,
                            "s1_name": sample.s1_name,
                            "caption": sample.caption,
                            "split": sample.split,
                            "latitude": sample.latitude,
                            "longitude": sample.longitude,
                        }
                    )
                    + "\n"
                )
        emit(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(rows):,} samples)")

    emit("")
    emit("Writing manifest with SHA256 hashes (streams 2.8 GB, please wait)...")
    manifest = write_manifest(
        MANIFEST,
        samples=samples,
        report=report,
        annotations_parquet=ANNOTATIONS,
        imagery_metadata=IMAGERY_METADATA,
        lmdb_dir=LMDB_DIR,
        country=COUNTRY,
        season=SEASON,
        seed=SEED,
        preprocessing_version=PREPROCESSING_VERSION,
    )
    emit(f"  manifest -> {MANIFEST.relative_to(REPO_ROOT)}")
    emit(f"  annotations sha256 : {manifest['annotations'].get('sha256')}")
    emit(f"  lmdb sha256        : {manifest['imagery'].get('lmdb_sha256')}")
    emit("")
    emit("PREPARATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
