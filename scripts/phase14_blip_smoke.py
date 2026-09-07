"""Phase 14 smoke test: real BLIP change_model inference through the new code path."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO))

from geospatial import load_display_image  # noqa: E402
from ml.inference import change_model_infer as c  # noqa: E402

print("availability:", c.availability().__dict__, flush=True)
t1, _ = load_display_image(str(REPO / "data/demo/demo_optical_t1.tif"), max_side=512)
t2, _ = load_display_image(str(REPO / "data/demo/demo_optical_t2.tif"), max_side=512)
pred = c.predict_pair(t1, t2, "Did anything change between the two images?")
print(
    "RESULT",
    {"answer": pred.answer, "raw": pred.raw, "confidence": pred.confidence,
     "convention": pred.convention, "device": pred.device, "version": pred.model_version},
    flush=True,
)
