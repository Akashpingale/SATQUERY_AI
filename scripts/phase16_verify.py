"""Verify both teammate BLIP checkpoints load and run real inference through the modules."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO))

from geospatial import load_display_image  # noqa: E402
from ml.inference import caption_model_infer as cap  # noqa: E402
from ml.inference import change_model_infer as chg  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


print("change_model SHA:", sha(REPO / "change_model" / "model.safetensors"), flush=True)
print("caption availability:", cap.availability().__dict__, flush=True)
print("change  availability:", chg.availability().__dict__, flush=True)

img, _ = load_display_image(str(REPO / "data/demo/demo_optical.tif"), max_side=512)
cpred = cap.predict(img)
print("CAPTION:", repr(cpred.caption), "conf=", cpred.confidence, "dev=", cpred.device, flush=True)

t1, _ = load_display_image(str(REPO / "data/demo/demo_optical_t1.tif"), max_side=512)
t2, _ = load_display_image(str(REPO / "data/demo/demo_optical_t2.tif"), max_side=512)
gpred = chg.predict_pair(t1, t2, "Did anything change between the two images?")
print("CHANGE:", repr(gpred.answer), "conf=", gpred.confidence, flush=True)
