"""Phase 15: prove the trained clip_blip checkpoint reloads in a fresh process and infers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO))

from geospatial import load_display_image  # noqa: E402
from ml.inference import cdvqa_transfer_infer as m  # noqa: E402

ckpt = REPO / "models/phase15/cdvqa_clip_blip/cdvqa_transfer_best.pt"
sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
print("availability:", m.availability().__dict__, flush=True)
print("checkpoint_sha256:", sha, flush=True)

t1, _ = load_display_image(str(REPO / "data/demo/demo_optical_t1.tif"), max_side=512)
t2, _ = load_display_image(str(REPO / "data/demo/demo_optical_t2.tif"), max_side=512)
pred = m.predict(t1, t2, "Did anything change between the two images?")
print("RESULT", json.dumps({
    "answer": pred.answer, "confidence": round(pred.confidence, 4),
    "top5": [(a, round(p, 4)) for a, p in pred.ranked],
    "version": pred.model_version, "device": pred.device,
}), flush=True)
