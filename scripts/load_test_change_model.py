"""Load + minimal real-inference test for the external `change_model/` checkpoint.

Loads with transformers (BLIP), measures load time / RAM / VRAM / latency, then runs REAL
inference on real remote-sensing imagery already in the project. Read-only w.r.t. the checkpoint.

    python scripts/load_test_change_model.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_DIR = REPO / "change_model"
OUT = REPO / "reports" / "change_model"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:  # noqa: BLE001 - psutil optional
        return -1.0


def main() -> int:
    import torch
    from PIL import Image
    from transformers import BlipForQuestionAnswering, BlipProcessor

    report: dict = {"model_dir": str(MODEL_DIR)}
    print(f"transformers import OK; torch {torch.__version__}")

    # --- load ------------------------------------------------------------------
    t0 = time.time()
    processor = BlipProcessor.from_pretrained(MODEL_DIR)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_DIR)
    model.eval()
    load_cpu = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    dtype = next(model.parameters()).dtype
    print(f"loaded on CPU in {load_cpu:.1f}s  params={n_params:,}  dtype={dtype}")
    report["load_cpu_seconds"] = round(load_cpu, 2)
    report["parameters"] = n_params
    report["dtype"] = str(next(model.parameters()).dtype)
    report["rss_after_load_mb"] = round(rss_mb(), 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.to(device)
        report["move_to_gpu_seconds"] = round(time.time() - t0, 2)
        report["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
        print(f"moved to CUDA in {report['move_to_gpu_seconds']}s  "
              f"VRAM allocated {report['vram_allocated_mb']} MB")
    report["device"] = device

    # --- real inference on real remote-sensing imagery -------------------------
    # SECOND aerial image (real RS data already provisioned) + demo optical GeoTIFF preview.
    second = Path(os.environ.get("SECOND_DIR", r"D:\dataset\second"))
    samples: list[tuple[str, Image.Image]] = []
    sar = second / "im1" / "00020.png"
    if sar.is_file():
        samples.append(("SECOND/im1/00020.png", Image.open(sar).convert("RGB")))
    demo = REPO / "data" / "demo" / "demo_optical.tif"
    if demo.is_file():
        from geospatial import load_display_image

        img, _ = load_display_image(demo, max_side=512)
        samples.append(("data/demo/demo_optical.tif", img))

    questions = [
        "What is in this image?",
        "Is there water in this image?",
        "Are there buildings in this image?",
        "How many roads are visible?",
    ]

    results = []
    for name, image in samples:
        for q in questions:
            inputs = processor(images=image, text=q, return_tensors="pt").to(device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=20)
            latency = time.time() - t0
            answer = processor.decode(out[0], skip_special_tokens=True)
            finite = bool(out.numel() > 0)
            print(f"  [{name}] Q: {q}\n      -> '{answer}'  ({latency:.2f}s)")
            results.append(
                {"image": name, "question": q, "answer": answer,
                 "latency_seconds": round(latency, 3), "valid_output": finite}
            )
    report["inference"] = results
    if results:
        report["mean_latency_seconds"] = round(
            sum(r["latency_seconds"] for r in results) / len(results), 3
        )
    if device == "cuda":
        report["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        print(f"peak VRAM {report['peak_vram_mb']} MB")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "load_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'load_test.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
