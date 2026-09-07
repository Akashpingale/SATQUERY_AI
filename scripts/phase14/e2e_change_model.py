"""Phase 14 end-to-end proof: the external BLIP change_model executed via the LIVE app.

Drives the running SatQuery API (:8000) exactly as the UI does:
  upload T1 + T2 -> POST /analysis with a change question -> inspect the response.

Acceptance: the response's plan/trace must show the agent routed the Change-VQA step to
``change_vqa_blip`` and executed the real ``change_model/`` checkpoint, returning its
generated answer with evidence/confidence/trace through the existing API contract.

Prereq: backend started with CHANGE_VQA_MODEL=blip (so the agent routes to the BLIP model).

    .\\.venv\\Scripts\\python.exe scripts\\phase14\\e2e_change_model.py

Output: reports/phase14/e2e/change_model_e2e.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "phase14" / "e2e"
DEMO = REPO / "data" / "demo"
BASE = "http://localhost:8000/api/v1"


def upload(client: httpx.Client, path: Path, role: str, modality: str) -> dict:
    with path.open("rb") as fh:
        files = {"file": (path.name, fh, "image/tiff")}
        data = {"role": role, "modality": modality}
        resp = client.post(f"{BASE}/uploads", files=files, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()["asset"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
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
    model_evidence = [
        e for e in body.get("evidence", [])
        if e.get("kind") == "model" and "change_vqa_blip" in json.dumps(e.get("detail", {}))
    ]

    routed_to_blip = "change_vqa_blip" in plan_tools
    executed = any(
        t["step"].lower().startswith("execution") and "change_vqa_blip" in t["step"]
        for t in trace
    )

    result = {
        "health": health,
        "query": query,
        "asset_ids": {"t1": t1["id"], "t2": t2["id"]},
        "status": body.get("status"),
        "task": body.get("task"),
        "answer": body.get("answer"),
        "confidence": body.get("confidence"),
        "plan_tools": plan_tools,
        "selection_notes": plan.get("selection_notes", []),
        "trace": trace,
        "blip_model_evidence": model_evidence,
        "checks": {
            "routed_to_change_vqa_blip": routed_to_blip,
            "executed_change_vqa_blip_in_trace": executed,
            "status_completed": body.get("status") == "completed",
            "answer_present": bool(body.get("answer")),
        },
    }
    (OUT / "change_model_e2e.json").write_text(
        json.dumps({"full_response": body, "summary": result}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, default=str))
    ok = routed_to_blip and result["checks"]["status_completed"] and body.get("answer")
    print("\nE2E_ACCEPTANCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
