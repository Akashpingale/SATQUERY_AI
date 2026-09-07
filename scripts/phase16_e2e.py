"""Phase 16 live E2E sanity: both trained BLIP models through the running SatQuery API.

Confirms captioning_blip returns a real generated caption (not the CLIP one, not an error),
and change_vqa_blip returns a real generated change answer, via /api/v1/analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "data" / "demo"
BASE = "http://localhost:8000/api/v1"
OUT = REPO / "reports" / "phase16"


def upload(client, path, role, modality):
    with path.open("rb") as fh:
        r = client.post(f"{BASE}/uploads", files={"file": (path.name, fh, "image/tiff")},
                        data={"role": role, "modality": modality}, timeout=120)
    r.raise_for_status()
    return r.json()["asset"]


def tool_of(body):
    steps = (body.get("plan") or {}).get("steps", [])
    return steps[-1]["tool_id"] if steps else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        health = client.get(f"{BASE}/health", timeout=30).json()

        opt = upload(client, DEMO / "demo_optical.tif", "single", "optical")
        cap = client.post(f"{BASE}/analysis",
                          json={"query": "Describe this scene.", "asset_ids": [opt["id"]]},
                          timeout=600).json()

        t1 = upload(client, DEMO / "demo_optical_t1.tif", "t1", "optical")
        t2 = upload(client, DEMO / "demo_optical_t2.tif", "t2", "optical")
        chg = client.post(f"{BASE}/analysis",
                          json={"query": "What changed between the two dates?",
                                "asset_ids": [t1["id"], t2["id"]]}, timeout=600).json()

    result = {
        "health": health,
        "captioning": {
            "tool": tool_of(cap), "task": cap.get("task"), "status": cap.get("status"),
            "answer": cap.get("answer"), "confidence": cap.get("confidence"),
            "check_uses_blip": tool_of(cap) == "captioning_blip",
        },
        "change_vqa": {
            "tool": tool_of(chg), "task": chg.get("task"), "status": chg.get("status"),
            "answer": chg.get("answer"), "confidence": chg.get("confidence"),
            "check_uses_blip": tool_of(chg) == "change_vqa_blip",
        },
    }
    (OUT / "e2e_sanity.json").write_text(json.dumps({"caption_full": cap, "change_full": chg,
                                                     "summary": result}, indent=2, default=str),
                                         encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    ok = result["captioning"]["check_uses_blip"] and result["change_vqa"]["check_uses_blip"] \
        and cap.get("status") == "completed" and chg.get("status") == "completed"
    print("\nE2E_SANITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
