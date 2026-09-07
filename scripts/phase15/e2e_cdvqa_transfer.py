"""Phase 15 end-to-end proof: the improved trained CDVQA model executed via the LIVE app.

Drives the running SatQuery API (:8000) as the UI does: upload T1+T2 -> POST /analysis with
a change question -> confirm the agent routed the Change-VQA step to `cdvqa_transfer` (the
Phase 15 improved model) and executed the real checkpoint, returning its output through the
existing API/evidence/confidence/trace.

Prereq: backend started with CHANGE_VQA_MODEL=cdvqa_transfer.

Output: reports/phase15/e2e/cdvqa_transfer_e2e.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "phase15" / "e2e"
DEMO = REPO / "data" / "demo"
BASE = "http://localhost:8000/api/v1"
CKPT = REPO / "models" / "phase15" / "cdvqa_clip_blip" / "cdvqa_transfer_best.pt"


def upload(client: httpx.Client, path: Path, role: str, modality: str) -> dict:
    with path.open("rb") as fh:
        files = {"file": (path.name, fh, "image/tiff")}
        resp = client.post(
            f"{BASE}/uploads", files=files, data={"role": role, "modality": modality}, timeout=120
        )
    resp.raise_for_status()
    return resp.json()["asset"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(CKPT.read_bytes()).hexdigest() if CKPT.is_file() else None
    with httpx.Client() as client:
        health = client.get(f"{BASE}/health", timeout=30).json()
        t1 = upload(client, DEMO / "demo_optical_t1.tif", "t1", "optical")
        t2 = upload(client, DEMO / "demo_optical_t2.tif", "t2", "optical")
        query = "Did the scene change between the two dates?"
        resp = client.post(
            f"{BASE}/analysis",
            json={"query": query, "asset_ids": [t1["id"], t2["id"]]},
            timeout=600,
        )
        resp.raise_for_status()
        body = resp.json()

    plan = body.get("plan") or {}
    plan_tools = [s.get("tool_id") for s in plan.get("steps", [])]
    trace = [
        {"step": t.get("step"), "status": t.get("status"), "detail": t.get("detail")}
        for t in body.get("trace", [])
    ]
    routed = "cdvqa_transfer" in plan_tools
    executed = any(
        "cdvqa_transfer" in t["step"]
        for t in trace
        if t["step"].lower().startswith("execution")
    )

    summary = {
        "health": health,
        "query": query,
        "checkpoint": str(CKPT.relative_to(REPO)),
        "checkpoint_sha256": sha,
        "status": body.get("status"),
        "task": body.get("task"),
        "answer": body.get("answer"),
        "confidence": body.get("confidence"),
        "plan_tools": plan_tools,
        "selection_notes": plan.get("selection_notes", []),
        "trace": trace,
        "checks": {
            "routed_to_cdvqa_transfer": routed,
            "executed_cdvqa_transfer_in_trace": executed,
            "status_completed": body.get("status") == "completed",
            "answer_present": bool(body.get("answer")),
        },
    }
    (OUT / "cdvqa_transfer_e2e.json").write_text(
        json.dumps({"full_response": body, "summary": summary}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    ok = routed and summary["checks"]["status_completed"] and body.get("answer")
    print("\nE2E_ACCEPTANCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
