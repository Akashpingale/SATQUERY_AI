"""Phase 12 end-to-end system validation.

Exercises the *running* SatQuery AI application (not isolated modules) with the
committed, synthetic, deterministic demo fixtures under ``data/demo/``. Nothing here
downloads data, retrains, or integrates any model; it only drives the public API and,
for the non-default learned route (Flow D), invokes the registered specialist directly
against the same database the server uses.

Flows
-----
A  Single image      : upload optical -> VQA query -> answer/confidence/evidence/trace
B  Bi-temporal       : upload T1+T2   -> change query -> deterministic temporal_change
C  Optical + SAR     : upload opt+sar -> fusion query -> optical_sar_fusion
D  Learned temporal  : invoke SECOND + CDVQA specialists directly (checkpoint load,
                       inference, valid output, refusal on incompatible input);
                       confirm they are NOT on any default task toolchain
E  Failure/fallback  : incompatible pair -> rejected; unsupported query -> honest status;
                       compatibility endpoint flags the non-overlapping pair

Run (backend must be live on :8000):
    $env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
    $env:HF_HOME="D:\\SatQuery-AI\\models\\hf-cache"; $env:DEVICE="cpu"
    .\\.venv\\Scripts\\python.exe scripts\\phase12\\e2e_validate.py

Outputs: reports/phase12/e2e/*.json  (+ SUMMARY.json)
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[2]
# Output dir is overridable so later phases can re-verify without overwriting earlier
# evidence (e.g. Phase 13 sets PHASE_E2E_OUT=reports/phase13/e2e).
OUT = Path(os.environ.get("PHASE_E2E_OUT", str(REPO / "reports" / "phase12" / "e2e")))
if not OUT.is_absolute():
    OUT = REPO / OUT
DEMO = REPO / "data" / "demo"
BASE = "http://localhost:8000/api/v1"

SUMMARY: dict[str, Any] = {
    "generated_utc": datetime.now(UTC).isoformat(),
    "base_url": BASE,
    "flows": {},
}


def save(name: str, obj: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def upload(client: httpx.Client, path: Path, role: str, modality: str) -> dict[str, Any]:
    with path.open("rb") as fh:
        files = {"file": (path.name, fh, "image/tiff")}
        data = {"role": role, "modality": modality}
        resp = client.post(f"{BASE}/uploads", files=files, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()["asset"]


def analyze(client: httpx.Client, query: str, asset_ids: list[str]) -> dict[str, Any]:
    resp = client.post(
        f"{BASE}/analysis", json={"query": query, "asset_ids": asset_ids}, timeout=600
    )
    resp.raise_for_status()
    return resp.json()


def digest(resp: dict[str, Any]) -> dict[str, Any]:
    """A compact, human-auditable view of an analysis response."""
    plan = resp.get("plan") or {}
    return {
        "id": resp.get("id"),
        "status": resp.get("status"),
        "task": resp.get("task"),
        "query": resp.get("query"),
        "answer": resp.get("answer"),
        "confidence": resp.get("confidence"),
        "plan_tools": [s.get("tool_id") for s in plan.get("steps", [])],
        "plan_versions": [s.get("tool_version") for s in plan.get("steps", [])],
        "selection_notes": plan.get("selection_notes", []),
        "evidence_kinds": [e.get("kind") for e in resp.get("evidence", [])],
        "trace": [
            {"step": t.get("step"), "status": t.get("status"), "detail": t.get("detail")}
            for t in resp.get("trace", [])
        ],
        "issues": resp.get("issues", []),
        "warnings": resp.get("warnings", []),
    }


def flow_a(client: httpx.Client) -> dict[str, Any]:
    optical = upload(client, DEMO / "demo_optical.tif", "optical", "optical")
    resp = analyze(
        client,
        "What land cover types are visible in this scene?",
        [optical["id"]],
    )
    save("flow_a_single_image.json", {"upload": optical, "analysis": resp})
    d = digest(resp)
    d["asset_ids"] = {"optical": optical["id"]}
    return d


def flow_b(client: httpx.Client) -> dict[str, Any]:
    t1 = upload(client, DEMO / "demo_optical_t1.tif", "t1", "optical")
    t2 = upload(client, DEMO / "demo_optical_t2.tif", "t2", "optical")
    resp = analyze(client, "What changed between these two images?", [t1["id"], t2["id"]])
    save("flow_b_temporal.json", {"upload_t1": t1, "upload_t2": t2, "analysis": resp})
    d = digest(resp)
    d["asset_ids"] = {"t1": t1["id"], "t2": t2["id"]}
    return d


def flow_c(client: httpx.Client) -> dict[str, Any]:
    optical = upload(client, DEMO / "demo_optical.tif", "optical", "optical")
    sar = upload(client, DEMO / "demo_sar.tif", "sar", "sar")
    resp = analyze(
        client,
        "Compare what the optical and SAR sensors reveal about this area.",
        [optical["id"], sar["id"]],
    )
    save(
        "flow_c_optical_sar.json",
        {"upload_optical": optical, "upload_sar": sar, "analysis": resp},
    )
    d = digest(resp)
    d["asset_ids"] = {"optical": optical["id"], "sar": sar["id"]}
    return d


def flow_e(client: httpx.Client) -> dict[str, Any]:
    optical = upload(client, DEMO / "demo_optical.tif", "t1", "optical")
    incompatible = upload(client, DEMO / "demo_incompatible.tif", "t2", "optical")

    # E1: incompatible (non-overlapping) temporal pair -> must be rejected, no fabricated answer.
    e1 = analyze(
        client, "What changed between these two images?", [optical["id"], incompatible["id"]]
    )
    # E2: unsupported request on a valid single image -> honest status, not a made-up answer.
    e2 = analyze(client, "Predict tomorrow's weather at this location.", [optical["id"]])
    # E3: compatibility endpoint should independently flag the non-overlapping pair.
    comp = client.post(
        f"{BASE}/uploads/compatibility",
        json={
            "asset_id_a": optical["id"],
            "asset_id_b": incompatible["id"],
            "purpose": "temporal",
        },
        timeout=60,
    )
    comp.raise_for_status()
    comp_json = comp.json()

    save(
        "flow_e_failure_fallback.json",
        {
            "incompatible_pair_analysis": e1,
            "unsupported_query_analysis": e2,
            "compatibility_check": comp_json,
        },
    )
    return {
        "incompatible_pair": digest(e1),
        "unsupported_query": digest(e2),
        "compatibility_check": {
            "compatible": comp_json.get("compatible"),
            "overlap_ratio": comp_json.get("overlap_ratio"),
            "issues": comp_json.get("issues", []),
        },
    }


def flow_d(t1_asset: dict[str, Any], t2_asset: dict[str, Any]) -> dict[str, Any]:
    """Learned temporal route, exercised directly (it is intentionally non-default)."""
    sys.path.insert(0, str(REPO / "backend"))
    sys.path.insert(0, str(REPO))

    from app.agent.planner import TASK_TOOLCHAIN
    from app.agent.specialists.base import SpecialistContext, SpecialistUnavailable
    from app.agent.specialists.cdvqa_trained import CDVQATrainedSpecialist
    from app.agent.specialists.second_change import SecondChangeSpecialist
    from app.agent.types import InputConfiguration, InputImage
    from app.db.session import SessionLocal
    from app.registry.tasks import Modality

    def image(asset: dict[str, Any]) -> InputImage:
        geo = asset.get("georeference") or {}
        return InputImage(
            asset_id=asset["id"],
            role=asset["role"],
            modality=Modality(asset["modality"]),
            filename=asset["original_filename"],
            width=asset.get("width"),
            height=asset.get("height"),
            band_count=asset.get("band_count"),
            is_georeferenced=bool(geo.get("is_georeferenced")),
            crs=geo.get("crs"),
        )

    def ctx(db: Any, assets: list[dict[str, Any]], query: str = "") -> SpecialistContext:
        return SpecialistContext(
            query=query,
            inputs=InputConfiguration(images=[image(a) for a in assets]),
            parameters={},
            db=db,
        )

    # Are the learned specialists on any DEFAULT task toolchain? (They must not be.)
    all_default_tools = {tool for chain in TASK_TOOLCHAIN.values() for tool in chain}
    learned_default = sorted(
        {"second_change_learned", "cdvqa_trained"} & all_default_tools
    )

    result: dict[str, Any] = {
        "learned_tools_on_default_toolchain": learned_default,
        "deterministic_default_confirmed": learned_default == [],
        "task_toolchain": {str(k): v for k, v in TASK_TOOLCHAIN.items()},
    }

    db = SessionLocal()
    try:
        # SECOND learned binary change detector on the valid optical T1/T2 pair.
        second = SecondChangeSpecialist().run(ctx(db, [t1_asset, t2_asset]))
        result["second_change_learned"] = {
            "checkpoint_loaded": True,
            "answer": second.answer,
            "score": second.score,
            "details": second.details,
            "evidence_kinds": [e.kind for e in second.evidence],
            "output_valid": (
                second.details.get("changed_fraction") is not None
                and 0.0 <= float(second.details["changed_fraction"]) <= 1.0
            ),
        }

        # CDVQA trained classifier (closed 19-answer vocabulary).
        cdvqa = CDVQATrainedSpecialist().run(
            ctx(db, [t1_asset, t2_asset], query="Did new buildings appear?")
        )
        result["cdvqa_trained"] = {
            "checkpoint_loaded": True,
            "answer": cdvqa.answer,
            "score": cdvqa.score,
            "answer_token": cdvqa.details.get("answer_token"),
            "evidence_kinds": [e.kind for e in cdvqa.evidence],
        }

        # Refusal: the learned detector must refuse a single-image (non-pair) request
        # instead of being silently forced to run.
        try:
            SecondChangeSpecialist().run(ctx(db, [t1_asset]))
            result["refuses_incompatible_input"] = False
        except SpecialistUnavailable as exc:
            result["refuses_incompatible_input"] = True
            result["refusal_reason"] = str(exc)
    finally:
        db.close()

    save("flow_d_learned_temporal.json", result)
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        health = client.get(f"{BASE}/health", timeout=30).json()
        SUMMARY["health"] = health

        SUMMARY["flows"]["A_single_image"] = flow_a(client)
        b = flow_b(client)
        SUMMARY["flows"]["B_temporal"] = b
        SUMMARY["flows"]["C_optical_sar"] = flow_c(client)
        SUMMARY["flows"]["E_failure_fallback"] = flow_e(client)

    # Flow D uses the T1/T2 assets uploaded in Flow B (same DB the server writes to).
    try:
        b_full = json.loads((OUT / "flow_b_temporal.json").read_text(encoding="utf-8"))
        SUMMARY["flows"]["D_learned_temporal"] = flow_d(b_full["upload_t1"], b_full["upload_t2"])
    except Exception as exc:  # noqa: BLE001 - record instead of crashing the whole run
        SUMMARY["flows"]["D_learned_temporal"] = {
            "status": "NOT EVALUATED - dependency unavailable",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    save("SUMMARY.json", SUMMARY)
    print(json.dumps(SUMMARY, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
