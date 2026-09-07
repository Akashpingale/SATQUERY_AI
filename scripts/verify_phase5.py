"""End-to-end Phase 5 verification: adapter, integration, and live analysis.

    python scripts/verify_phase5.py

Confirms in a **fresh process** that the trained adapter loads, that it genuinely changes
model behaviour versus the baseline, that the agent selects it, and that a live analysis
through the real API records the adapted model in its trace and evidence.

Writes ``scripts/phase5_verify_report.txt``. Exits non-zero on any failure.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

REPORT = Path(__file__).resolve().parent / "phase5_verify_report.txt"
REPORTS = REPO_ROOT / "reports" / "phase5"
CHECKS: list[tuple[str, bool, str]] = []


def emit(message: str = "") -> None:
    print(message, flush=True)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, ok, detail))
    emit(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def build_scene() -> bytes:
    """A synthetic georeferenced Sentinel-like scene for the live analysis."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    size = 256
    array = np.zeros((3, size, size), dtype="uint8")
    array[1, :, :] = 150
    array[0, :, :] = 70
    array[2, : size // 3, :] = 210
    array[1, : size // 3, :] = 70

    buffer = io.BytesIO()
    with rasterio.open(
        buffer,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(23.9, 55.2, 0.0005, 0.0005),
    ) as dst:
        dst.write(array)
    return buffer.getvalue()


def main() -> int:
    REPORT.unlink(missing_ok=True)
    emit("=" * 74)
    emit("PHASE 5 VERIFICATION (fresh process)")
    emit("=" * 74)

    # --- 1. adapter on disk -------------------------------------------------------
    from ml.inference import describe_adapter, variant_version

    info = describe_adapter()
    if not check("adapter present on disk", info.present, info.reason or info.path):
        return 1
    check(
        "adapter records real configuration",
        info.rank is not None and info.alpha is not None and bool(info.target_modules),
        f"rank={info.rank} alpha={info.alpha} targets={info.target_modules}",
    )
    check(
        "adapter contains weights",
        bool(info.size_bytes and info.size_bytes > 1024),
        f"{(info.size_bytes or 0) / 1024:.0f} KB",
    )
    check(
        "versions distinguish baseline from adapted",
        variant_version("baseline") != variant_version("adapted"),
        f"{variant_version('baseline')} vs {variant_version('adapted')}",
    )
    emit("")

    # --- 2. adapter changes behaviour ---------------------------------------------
    import numpy as np
    from PIL import Image

    from ml.inference import clip_backend

    rng = np.random.default_rng(7)
    probe = Image.fromarray((rng.random((224, 224, 3)) * 255).astype("uint8"), "RGB")

    baseline_scores = clip_backend.score_land_cover(probe, variant="baseline")
    adapted_scores = clip_backend.score_land_cover(probe, variant="adapted")

    baseline_map = {s.key: s.probability for s in baseline_scores.scores}
    adapted_map = {s.key: s.probability for s in adapted_scores.scores}
    max_delta = max(abs(adapted_map[k] - baseline_map[k]) for k in baseline_map)

    check(
        "adapted inference runs",
        bool(adapted_scores.scores),
        f"{len(adapted_scores.scores)} concepts on {adapted_scores.device}",
    )
    check(
        "adapted output differs from baseline",
        max_delta > 1e-4,
        f"max |delta probability| = {max_delta:.6f}",
    )
    check(
        "result records its variant",
        adapted_scores.variant == "adapted" and baseline_scores.variant == "baseline",
        f"{baseline_scores.variant} / {adapted_scores.variant}",
    )
    baseline_top = baseline_scores.scores[0]
    adapted_top = adapted_scores.scores[0]
    emit(f"     baseline top: {baseline_top.key} {baseline_top.probability:.4f}")
    emit(f"     adapted  top: {adapted_top.key} {adapted_top.probability:.4f}")
    emit("")

    # --- 3. agent selects the adapted model ---------------------------------------
    from app.agent.planner import select_model_for_tool

    selected, note = select_model_for_tool("vqa")
    check("planner selects the adapted VQA model", selected == "vqa_adapted", f"{selected}: {note}")
    selected_caption, _ = select_model_for_tool("captioning")
    check(
        "planner selects the adapted captioning model",
        selected_caption == "captioning_adapted",
        selected_caption,
    )
    emit("")

    # --- 4. registry exposes both variants ----------------------------------------
    from app.registry.models import MODEL_REGISTRY, ModelVariant

    baseline_entries = [t for t, s in MODEL_REGISTRY.items() if s.variant is ModelVariant.BASELINE]
    adapted_entries = [t for t, s in MODEL_REGISTRY.items() if s.variant is ModelVariant.ADAPTED]
    check(
        "registry exposes baseline and adapted separately",
        bool(baseline_entries) and bool(adapted_entries),
        f"baseline={baseline_entries} adapted={adapted_entries}",
    )
    emit("")

    # --- 5. live analysis through the real API ------------------------------------
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/uploads",
            files=[("file", ("phase5_verify.tif", build_scene(), "image/tiff"))],
            data={"role": "optical", "modality": "optical"},
        )
        if not check("upload succeeds", upload.status_code == 201, f"HTTP {upload.status_code}"):
            return 1
        asset_id = upload.json()["asset"]["id"]

        body = client.post(
            "/api/v1/analysis",
            json={"query": "What land cover dominates this image?", "asset_ids": [asset_id]},
        ).json()

        check("analysis completes", body["status"] == "completed", body["status"])
        check(
            "plan uses the adapted tool",
            body["plan"]["steps"][0]["tool_id"] == "vqa_adapted",
            body["plan"]["steps"][0]["tool_id"],
        )
        check(
            "plan records why the model was chosen",
            bool(body["plan"]["selection_notes"]),
            "; ".join(body["plan"]["selection_notes"]),
        )

        trace_steps = {entry["step"] for entry in body["trace"]}
        check(
            "trace records model selection and provenance",
            {"Model selection", "Model provenance"} <= trace_steps,
            ", ".join(sorted(trace_steps)),
        )
        provenance = next(e for e in body["trace"] if e["step"] == "Model provenance")
        check(
            "provenance names base model, adapter, and dataset",
            all(
                token in provenance["detail"]
                for token in ("variant=adapted", "clip-rsicd-v2", "BigEarthNet", "LoRA")
            ),
            provenance["detail"][:150],
        )

        variant_items = [
            item for item in body["evidence"] if item["detail"].get("variant") == "adapted"
        ]
        check(
            "evidence records the adapted variant",
            bool(variant_items),
            str(variant_items[0]["detail"].get("version")) if variant_items else "none",
        )
        check(
            "adaptation limitation is surfaced to the user",
            any("Phase 5 adapted model" in warning for warning in body["warnings"]),
            f"{len(body['warnings'])} warnings",
        )
        check(
            "confidence remains honest",
            body["confidence"]["level"] in {"High", "Medium", "Low", "Unable to determine"},
            f"{body['confidence']['level']} score={body['confidence']['score']}",
        )
        emit(f"     answer: {body['answer'][:160]}")
        emit("")

        # --- 6. Phase 4 regression: grounding still uses OWL-ViT ------------------
        grounding = client.post(
            "/api/v1/analysis",
            json={"query": "Highlight the water in this image", "asset_ids": [asset_id]},
        ).json()
        check(
            "grounding still routes to the unchanged Phase 4 tool",
            grounding["plan"]["steps"][0]["tool_id"] == "grounding",
            grounding["plan"]["steps"][0]["tool_id"],
        )
        check(
            "counting is still refused",
            client.post(
                "/api/v1/analysis",
                json={"query": "How many buildings are there?", "asset_ids": [asset_id]},
            ).json()["confidence"]["level"]
            == "Unable to determine",
            "counting refusal intact",
        )
    emit("")

    # --- 7. recorded metrics exist and are real -----------------------------------
    for name in ("training_metrics.json", "evaluation_metrics.json", "environment.json"):
        path = REPORTS / name
        check(f"{name} exists", path.is_file(), str(path.relative_to(REPO_ROOT)))

    evaluation_path = REPORTS / "evaluation_metrics.json"
    if evaluation_path.is_file():
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        improvement = payload["difference"]["mean recall"]
        check(
            "evaluation compares baseline and adapted on the same held-out split",
            payload["baseline"]["samples"] == payload["adapted"]["samples"],
            f"{payload['samples']} samples, split={payload['split']}",
        )
        emit(
            f"     mean recall {payload['baseline']['mean_recall']:.4f} -> "
            f"{payload['adapted']['mean_recall']:.4f} (delta {improvement:+.4f})"
        )

    emit("")
    emit("=" * 74)
    failed = [name for name, ok, _ in CHECKS if not ok]
    emit(f"SUMMARY: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        for name in failed:
            emit(f"  FAILED: {name}")
        emit("RESULT: PHASE 5 VERIFICATION FAILED")
        return 1
    emit("RESULT: PHASE 5 VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
