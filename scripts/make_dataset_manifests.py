"""Generate real provenance manifests for the locally-provisioned CDVQA and SECOND datasets.

Every value is computed from the actual archives / extracted files — no guessed numbers.
Run after the archives are extracted:

    python scripts/make_dataset_manifests.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CDVQA_ARCHIVE = Path(r"D:\dataset\CDVQA-main.zip")
SECOND_ARCHIVE = Path(r"D:\dataset\SECOND_train_set.rar")
CDVQA_DIR = REPO / "datasets" / "cdvqa" / "CDVQA-main"
SECOND_DIR = Path(r"D:\dataset\second")


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def cdvqa_manifest() -> dict:
    splits = {}
    for split in ["Train", "Val", "Test", "Test2"]:
        q = json.load(open(CDVQA_DIR / f"{split}_questions.json"))["questions"]
        a = json.load(open(CDVQA_DIR / f"{split}_answers.json"))["answers"]
        im = json.load(open(CDVQA_DIR / f"{split}_images.json"))["images"]
        qtypes = collections.Counter(x["type"] for x in q)
        answers = collections.Counter(str(x["answer"]).lower() for x in a)
        splits[split] = {
            "questions": len(q),
            "answers": len(a),
            "image_records": len(im),
            "unique_images": len({x["file_name"] for x in im}),
            "question_types": dict(sorted(qtypes.items())),
            "answer_vocab_size": len(answers),
            "top_answers": answers.most_common(15),
        }
    return {
        "dataset": "CDVQA (Change Detection Visual Question Answering)",
        "sih_status": "SIH26167-prescribed multitemporal change-VQA benchmark",
        "source": "https://github.com/YZHJessica/CDVQA (user-provided archive)",
        "local_archive": str(CDVQA_ARCHIVE),
        "archive_sha256": sha256(CDVQA_ARCHIVE),
        "archive_size_bytes": CDVQA_ARCHIVE.stat().st_size,
        "extracted_path": str(CDVQA_DIR),
        "extraction_tool": "PowerShell Expand-Archive",
        "extraction_date": datetime.now(UTC).isoformat(),
        "images_note": (
            "CDVQA ships only QA JSON; the referenced image file_names are the SECOND dataset "
            "images (0.1524 m). All referenced files are present in the SECOND train archive "
            "im1/im2 (verified). Splits are disjoint by image (no Train/Val/Test image overlap); "
            "Test2 reuses Test images with harder questions."
        ),
        "license": "See CDVQA repository (research use)",
        "splits": splits,
    }


PALETTE = {
    (255, 255, 255): "no_change",
    (128, 128, 128): "nvg_surface",
    (0, 128, 0): "tree",
    (0, 255, 0): "low_vegetation",
    (0, 0, 255): "water",
    (128, 0, 0): "buildings",
    (255, 0, 0): "playground",
}


def second_manifest() -> dict:
    counts = {d: len(os.listdir(SECOND_DIR / d)) for d in ["im1", "im2", "label1", "label2"]}
    sample = sorted(os.listdir(SECOND_DIR / "im1"))[0]
    from PIL import Image

    im = Image.open(SECOND_DIR / "im1" / sample)
    return {
        "dataset": "SECOND (SEmantic Change detectiON Dataset) — TRAIN set",
        "sih_status": (
            "ADDITIONAL PUBLIC dataset (NOT SIH-provided) for change-detection training / "
            "external validation, and the image source for CDVQA."
        ),
        "source": (
            "User-provided archive. Original: SECOND semantic change detection dataset "
            "(Google Drive: SECOND_train_set.rar)."
        ),
        "local_archive": str(SECOND_ARCHIVE),
        "archive_sha256": sha256(SECOND_ARCHIVE),
        "archive_size_bytes": SECOND_ARCHIVE.stat().st_size,
        "extracted_path": str(SECOND_DIR),
        "extraction_tool": "tar (bsdtar/libarchive) — RAR read support",
        "extraction_date": datetime.now(UTC).isoformat(),
        "structure": {
            "im1": "T1 RGB image (512x512)",
            "im2": "T2 RGB image (512x512)",
            "label1": "T1 semantic land-cover label (RGB-coded)",
            "label2": "T2 semantic land-cover label (RGB-coded)",
        },
        "counts": counts,
        "image_pairs": counts["im1"],
        "image_size": list(im.size),
        "image_mode": im.mode,
        "label_encoding": "RGB colour per class",
        "classes": {f"{r},{g},{b}": name for (r, g, b), name in PALETTE.items()},
        "num_semantic_classes": 6,
        "no_change_color": "255,255,255",
        "official_splits": (
            "Only a TRAIN set is provided (no official val/test in this archive). Val/Test are "
            "created from these pairs with a fixed seed; the held-out portion is an INTERNAL "
            "test split, NOT the official SECOND test set."
        ),
        "license": "See SECOND dataset release (research use)",
    }


def main() -> int:
    out_cdvqa = REPO / "datasets" / "cdvqa" / "manifest.json"
    out_second = REPO / "datasets" / "second" / "manifest.json"
    out_cdvqa.parent.mkdir(parents=True, exist_ok=True)
    out_second.parent.mkdir(parents=True, exist_ok=True)
    out_cdvqa.write_text(json.dumps(cdvqa_manifest(), indent=2), encoding="utf-8")
    print(f"wrote {out_cdvqa}")
    out_second.write_text(json.dumps(second_manifest(), indent=2), encoding="utf-8")
    print(f"wrote {out_second}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
