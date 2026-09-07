"""Forensic inspection of the externally-supplied checkpoint in `change_model/`.

Read-only: computes SHA256 for every file and parses the safetensors header (tensor names,
shapes, dtypes, parameter count) WITHOUT loading weights into memory. Nothing is modified.

    python scripts/inspect_change_model.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "change_model"
OUT = REPO / "reports" / "change_model"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def read_safetensors_header(path: Path) -> dict:
    """Parse the safetensors JSON header only (first 8 bytes = little-endian header length)."""
    with path.open("rb") as h:
        (length,) = struct.unpack("<Q", h.read(8))
        header = json.loads(h.read(length).decode("utf-8"))
    return header


def main() -> int:
    if not MODEL_DIR.is_dir():
        print(f"MODEL DIR NOT FOUND: {MODEL_DIR}")
        return 1

    files = sorted(p for p in MODEL_DIR.iterdir() if p.is_file())
    print(f"=== files in {MODEL_DIR} ===")
    hashes = {}
    for f in files:
        digest = sha256(f)
        hashes[f.name] = {"size_bytes": f.stat().st_size, "sha256": digest}
        print(f"  {f.name:26} {f.stat().st_size:>13,} B  sha256={digest}")

    weights = MODEL_DIR / "model.safetensors"
    header = read_safetensors_header(weights)
    meta = header.pop("__metadata__", None)
    print(f"\n=== safetensors header ===\n  tensors: {len(header)}")
    print(f"  __metadata__: {meta!r}")

    total_params = 0
    dtypes = collections.Counter()
    prefixes = collections.Counter()
    for name, spec in header.items():
        shape = spec.get("shape", [])
        n = 1
        for dim in shape:
            n *= dim
        total_params += n
        dtypes[spec.get("dtype")] += 1
        prefixes[name.split(".")[0]] += 1

    print(f"  total parameters: {total_params:,}")
    print(f"  dtypes: {dict(dtypes)}")
    print("  top-level modules:")
    for prefix, count in prefixes.most_common():
        print(f"    {prefix:30} {count} tensors")

    # Head/task-revealing tensors: what output layers exist tells us the real task.
    print("\n=== task-revealing tensors (heads / classifiers / decoders) ===")
    keys = list(header)
    for name in keys:
        low = name.lower()
        markers = ("head", "classifier", "score", "logit", "seg", "mask", "decoder.cls")
        if any(t in low for t in markers):
            print(f"    {name}  shape={header[name]['shape']} dtype={header[name]['dtype']}")

    print("\n=== first / last 10 tensor names ===")
    for name in keys[:10]:
        print(f"    {name} {header[name]['shape']}")
    print("    ...")
    for name in keys[-10:]:
        print(f"    {name} {header[name]['shape']}")

    # Bi-temporal evidence check: does any tensor imply 2-image / 6-channel input?
    patch_embed = [n for n in keys if "patch_embed" in n.lower() and "weight" in n.lower()]
    print("\n=== input-shape evidence (patch embedding) ===")
    for name in patch_embed:
        print(f"    {name} shape={header[name]['shape']}  (in-channels = index 1)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoint_inventory.json").write_text(
        json.dumps(
            {
                "model_dir": str(MODEL_DIR),
                "files": hashes,
                "safetensors": {
                    "tensor_count": len(header),
                    "total_parameters": total_params,
                    "dtypes": dict(dtypes),
                    "metadata": meta,
                    "top_level_modules": dict(prefixes),
                    "tensor_names": keys,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT / 'checkpoint_inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
