"""Verify that the Phase 4 baseline models load and run on CPU.

Run from the repository root:

    python scripts/check_models.py

Writes a report to ``scripts/model_check_report.txt`` as well as stdout (the report file
is the reliable copy: some terminals truncate long-running output).

Reports real load times, real inference latency, and real model scores - nothing here is
simulated. Downloads go to the Hugging Face cache; point ``HF_HOME`` at a drive with free
space (see ``.env.example``).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT = Path(__file__).resolve().parent / "model_check_report.txt"


def emit(message: str) -> None:
    print(message, flush=True)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    REPORT.unlink(missing_ok=True)

    import numpy as np
    import torch
    import transformers
    from PIL import Image

    emit(
        f"torch {torch.__version__} | cuda={torch.cuda.is_available()} | "
        f"threads={torch.get_num_threads()}"
    )
    emit(f"transformers {transformers.__version__}")
    emit(f"HF cache: {os.environ.get('HF_HOME', '(default)')}")

    rng = np.random.default_rng(0)
    image = Image.fromarray((rng.random((256, 256, 3)) * 255).astype("uint8"), "RGB")

    failures = 0

    # --- CLIP fine-tuned on remote-sensing imagery --------------------------------
    try:
        from transformers import CLIPModel, CLIPProcessor

        start = time.time()
        clip = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")
        clip_processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")
        clip.eval()
        params = sum(p.numel() for p in clip.parameters()) / 1e6
        emit(f"CLIP-RSICD loaded in {time.time() - start:.1f}s | {params:.1f}M params")

        prompts = [
            "an aerial photo of water",
            "an aerial photo of dense buildings",
            "an aerial photo of farmland",
        ]
        start = time.time()
        with torch.no_grad():
            inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True)
            probs = clip(**inputs).logits_per_image.softmax(dim=-1)[0].tolist()
        emit(f"CLIP inference {time.time() - start:.2f}s | probs {[round(p, 4) for p in probs]}")
    except Exception as exc:  # noqa: BLE001 - this script reports, it does not raise
        emit(f"CLIP FAILED: {type(exc).__name__}: {exc}")
        failures += 1

    # --- OWL-ViT open-vocabulary detection ----------------------------------------
    try:
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        start = time.time()
        owl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        owl.eval()
        params = sum(p.numel() for p in owl.parameters()) / 1e6
        emit(f"OWL-ViT loaded in {time.time() - start:.1f}s | {params:.1f}M params")

        start = time.time()
        with torch.no_grad():
            inputs = owl_processor(
                text=[["a building", "a body of water"]], images=image, return_tensors="pt"
            )
            outputs = owl(**inputs)
            results = owl_processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=torch.tensor([[256, 256]]), threshold=0.05
            )
        emit(
            f"OWL-ViT inference {time.time() - start:.2f}s | "
            f"{len(results[0]['scores'])} detection(s) above 0.05"
        )
    except Exception as exc:  # noqa: BLE001
        emit(f"OWL-ViT FAILED: {type(exc).__name__}: {exc}")
        failures += 1

    summary = (
        "all baseline models load and run on CPU."
        if failures == 0
        else f"{failures} model(s) failed."
    )
    emit(f"RESULT: {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
