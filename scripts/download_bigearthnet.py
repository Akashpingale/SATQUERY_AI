"""Download the BigEarthNet.txt annotations and the Sentinel image subset used for Phase 5.

    python scripts/download_bigearthnet.py

Downloads (~2.8 GB total) into ``data/bigearthnet/``:

1. ``BigEarthNet.txt.parquet`` from the **official** Hugging Face repository
   ``BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`` (CDLA-Permissive-1.0). These are the
   image-text annotations the SIH requirement names.

2. ``hackelle/BigEarthNetV2-Lithuania-Summer-LMDB`` (CDLA-Permissive-1.0) - Sentinel-1 and
   Sentinel-2 pixels for the Lithuania/Summer subset, pre-converted to LMDB with
   ``rico-hdl`` (the officially recommended converter).

   PROVENANCE NOTE: repository 2 self-identifies as an **unofficial** pre-conversion of the
   official Zenodo release (10.5281/zenodo.10891137). It is used because the official
   imagery is 109.88 GB compressed (~170-220 GB extracted, plus another copy after LMDB
   conversion), which does not fit the available disk or the 4 GB GPU's useful working set.
   The preparation step verifies every downloaded patch id against the OFFICIAL annotation
   parquet, so image-text pairing is validated against the official source regardless.

Nothing here is committed to Git; ``data/`` is ignored.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "bigearthnet"
REPORT = Path(__file__).resolve().parent / "bigearthnet_download_report.txt"

ANNOTATIONS_REPO = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
ANNOTATION_FILES = ["BigEarthNet.txt.parquet", "ben_txt_datamodule.py"]

IMAGERY_REPO = "hackelle/BigEarthNetV2-Lithuania-Summer-LMDB"
IMAGERY_FILES = [
    "BENv2_lithuania_summer.lmdb/data.mdb",
    "BENv2_lithuania_summer.lmdb/lock.mdb",
    "metadata_lithuania_summer.parquet",
    "README.md",
]

REQUIRED_FREE_GB = 8.0


def emit(message: str = "") -> None:
    print(message, flush=True)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def gb(value: float) -> str:
    return f"{value / 1024 ** 3:.2f} GB"


def main() -> int:
    REPORT.unlink(missing_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(DEST.drive + "\\").free
    emit(f"destination : {DEST}")
    emit(f"free space  : {gb(free)}")
    if free < REQUIRED_FREE_GB * 1024**3:
        emit(f"ABORT: need at least {REQUIRED_FREE_GB} GB free.")
        return 1

    from huggingface_hub import hf_hub_download

    total = 0
    for repo_id, files, subdir in (
        (ANNOTATIONS_REPO, ANNOTATION_FILES, "annotations"),
        (IMAGERY_REPO, IMAGERY_FILES, "imagery"),
    ):
        emit("")
        emit("=" * 74)
        emit(f"{repo_id}  ->  {subdir}/")
        emit("=" * 74)
        target_dir = DEST / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=target_dir,
                )
            except Exception as exc:  # noqa: BLE001 - report, do not crash
                emit(f"  FAILED {filename}: {type(exc).__name__}: {exc}")
                return 2
            size = Path(path).stat().st_size
            total += size
            emit(f"  ok {filename}  ({size / 1024 / 1024:.1f} MB)")

    emit("")
    emit(f"TOTAL DOWNLOADED: {gb(total)}")
    emit("DOWNLOAD COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
