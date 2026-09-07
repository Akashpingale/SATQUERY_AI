"""Inspect the real BigEarthNet.txt Parquet schema and distributions.

    python scripts/inspect_bigearthnet_txt.py

Reports only measured values: actual schema, actual row count, actual category/split
distributions, and the actual overlap with the downloaded imagery subset. Nothing is
assumed from the dataset card.

Memory-safe: reads Parquet metadata first, then projects only the columns it needs, so a
445 MB / 9.6 M-row file is inspected without loading it all.

Writes ``scripts/bigearthnet_txt_report.txt``.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data" / "bigearthnet"
ANNOTATIONS = DATA / "annotations" / "BigEarthNet.txt.parquet"
IMAGERY_METADATA = DATA / "imagery" / "metadata_lithuania_summer.parquet"
LMDB_DIR = DATA / "imagery" / "BENv2_lithuania_summer.lmdb"

REPORT = Path(__file__).resolve().parent / "bigearthnet_txt_report.txt"


def emit(message: str = "") -> None:
    print(message, flush=True)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    REPORT.unlink(missing_ok=True)

    import pyarrow.parquet as pq

    if not ANNOTATIONS.is_file():
        emit(f"MISSING: {ANNOTATIONS}")
        emit("Run scripts/download_bigearthnet.py first.")
        return 1

    # --- 1. schema and size, from metadata only ----------------------------------
    parquet = pq.ParquetFile(ANNOTATIONS)
    metadata = parquet.metadata
    emit("=" * 74)
    emit("1. BigEarthNet.txt.parquet STRUCTURE (measured)")
    emit("=" * 74)
    emit(f"  file            : {ANNOTATIONS.name}")
    emit(f"  size on disk    : {ANNOTATIONS.stat().st_size / 1024 ** 2:.1f} MB")
    emit(f"  rows            : {metadata.num_rows:,}")
    emit(f"  columns         : {metadata.num_columns}")
    emit(f"  row groups      : {metadata.num_row_groups}")
    emit("  schema:")
    for field in parquet.schema_arrow:
        emit(f"    - {field.name}: {field.type}")

    # --- 2. sample rows -----------------------------------------------------------
    emit()
    emit("=" * 74)
    emit("2. SAMPLE ROWS (first row group, first 3 rows)")
    emit("=" * 74)
    head = parquet.read_row_group(0).slice(0, 3).to_pylist()
    for index, row in enumerate(head, start=1):
        emit(f"  --- row {index} ---")
        for key, value in row.items():
            text = str(value)
            if len(text) > 160:
                text = text[:157] + "..."
            emit(f"    {key}: {text}")

    # --- 3. distributions over the small categorical columns ---------------------
    emit()
    emit("=" * 74)
    emit("3. DISTRIBUTIONS (full scan of categorical columns)")
    emit("=" * 74)
    counters: dict[str, Counter] = {
        "type": Counter(),
        "category": Counter(),
        "split": Counter(),
        "season": Counter(),
    }
    country_counter: Counter = Counter()
    for batch in parquet.iter_batches(
        batch_size=250_000, columns=["type", "category", "split", "season", "country"]
    ):
        data = batch.to_pydict()
        for column in counters:
            counters[column].update(data[column])
        country_counter.update(data["country"])

    for column, counter in counters.items():
        emit(f"  {column} ({len(counter)} distinct):")
        for value, count in counter.most_common(20):
            emit(f"    {str(value):28s} {count:>10,}")
    emit(f"  country ({len(country_counter)} distinct), top 10:")
    for value, count in country_counter.most_common(10):
        emit(f"    {str(value):28s} {count:>10,}")

    # --- 4. the Lithuania/Summer slice we can actually pair with imagery ---------
    emit()
    emit("=" * 74)
    emit("4. LITHUANIA / SUMMER SLICE (pairs with the downloaded imagery)")
    emit("=" * 74)
    slice_splits: Counter = Counter()
    slice_types: Counter = Counter()
    slice_patches: set[str] = set()
    slice_s1: set[str] = set()
    rows_in_slice = 0
    for batch in parquet.iter_batches(
        batch_size=250_000,
        columns=["patch_id", "s1_name", "type", "split", "country", "season"],
    ):
        data = batch.to_pydict()
        for i in range(len(data["country"])):
            if data["country"][i] == "Lithuania" and data["season"][i] == "Summer":
                rows_in_slice += 1
                slice_splits[data["split"][i]] += 1
                slice_types[data["type"][i]] += 1
                slice_patches.add(data["patch_id"][i])
                slice_s1.add(data["s1_name"][i])
    emit(f"  annotations      : {rows_in_slice:,}")
    emit(f"  unique S2 patches: {len(slice_patches):,}")
    emit(f"  unique S1 names  : {len(slice_s1):,}")
    emit(f"  by split         : {dict(slice_splits)}")
    emit(f"  by type          : {dict(slice_types)}")

    # --- 5. verify the imagery actually matches the official annotations ---------
    emit()
    emit("=" * 74)
    emit("5. IMAGE/TEXT PAIRING VERIFICATION against the OFFICIAL annotations")
    emit("=" * 74)
    if not IMAGERY_METADATA.is_file():
        emit(f"  MISSING imagery metadata: {IMAGERY_METADATA}")
        return 1

    import pandas as pd

    imagery = pd.read_parquet(IMAGERY_METADATA)
    emit(f"  imagery metadata rows : {len(imagery):,}")
    emit(f"  imagery columns       : {list(imagery.columns)}")
    image_patches = set(imagery["patch_id"].astype(str))
    image_s1 = set(imagery["s1_name"].astype(str))
    emit(f"  imagery S2 patches    : {len(image_patches):,}")
    emit(f"  imagery S1 patches    : {len(image_s1):,}")

    both = slice_patches & image_patches
    emit(f"  S2 ids in BOTH        : {len(both):,}")
    emit(f"  annotation-only S2 ids: {len(slice_patches - image_patches):,}")
    emit(f"  imagery-only S2 ids   : {len(image_patches - slice_patches):,}")
    overlap = 100.0 * len(both) / max(1, len(image_patches))
    emit(f"  imagery covered by official annotations: {overlap:.2f}%")

    # --- 6. LMDB keys are really present -----------------------------------------
    emit()
    emit("=" * 74)
    emit("6. LMDB CONTENTS")
    emit("=" * 74)
    if not (LMDB_DIR / "data.mdb").is_file():
        emit(f"  MISSING LMDB: {LMDB_DIR}")
        return 1
    emit(f"  data.mdb size : {(LMDB_DIR / 'data.mdb').stat().st_size / 1024 ** 2:.1f} MB")

    import lmdb
    from safetensors.numpy import load as safetensor_load

    env = lmdb.open(str(LMDB_DIR), readonly=True, lock=False, readahead=False, max_dbs=1)
    with env.begin(write=False) as txn:
        emit(f"  entries       : {txn.stat()['entries']:,}")
        sample_patch = sorted(both)[0] if both else sorted(image_patches)[0]
        raw = txn.get(sample_patch.encode())
        if raw is None:
            emit(f"  KEY MISSING for {sample_patch}")
            return 1
        tensors = safetensor_load(raw)
        emit(f"  sample S2 key : {sample_patch}")
        emit(f"  S2 bands      : {sorted(tensors.keys())}")
        first = tensors[sorted(tensors.keys())[0]]
        emit(f"  band shape    : {first.shape}, dtype={first.dtype}")
        emit(f"  value range   : {first.min()} .. {first.max()}")

        s1_name = str(imagery.loc[imagery["patch_id"] == sample_patch, "s1_name"].iloc[0])
        raw_s1 = txn.get(s1_name.encode())
        if raw_s1 is not None:
            s1_tensors = safetensor_load(raw_s1)
            emit(f"  sample S1 key : {s1_name}")
            emit(f"  S1 bands      : {sorted(s1_tensors.keys())}")
        else:
            emit(f"  S1 key MISSING for {s1_name}")
    env.close()

    emit()
    emit("INSPECTION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
