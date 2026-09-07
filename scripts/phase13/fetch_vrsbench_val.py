"""Phase 13 - provision the official VRSBench validation images.

Downloads `Images_val.zip` from the official public HuggingFace dataset
(`xiang709/VRSBench`) into `D:\\dataset\\vrsbench\\`. This is the ONLY missing asset
that blocked VRSBench grounding in Phases 11/12. No annotations are downloaded (the
official `VRSBench_EVAL_referring.json` is already present locally); nothing is trained.

Resumable via huggingface_hub. Prints size + SHA256 on completion.

Run (network required):
    $env:HF_HUB_OFFLINE="0"; $env:TRANSFORMERS_OFFLINE="0"
    .\\.venv\\Scripts\\python.exe scripts\\phase13\\fetch_vrsbench_val.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

REPO_ID = "xiang709/VRSBench"
FILENAME = "Images_val.zip"
DEST_DIR = Path(r"D:\dataset\vrsbench")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    dest = DEST_DIR / FILENAME
    if dest.is_file() and dest.stat().st_size > 3_000_000_000:
        print(f"Already present: {dest} ({dest.stat().st_size/1048576:.1f} MB)")
    else:
        from huggingface_hub import hf_hub_download

        print(f"Downloading {REPO_ID}/{FILENAME} (dataset) ...", flush=True)
        cached = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
            local_dir=str(DEST_DIR),
        )
        cached_path = Path(cached)
        if cached_path.resolve() != dest.resolve():
            shutil.copy2(cached_path, dest)
        print(f"Downloaded -> {dest}", flush=True)

    size_mb = dest.stat().st_size / 1048576
    print(f"size_mb={size_mb:.2f}", flush=True)
    print(f"sha256={sha256(dest)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
