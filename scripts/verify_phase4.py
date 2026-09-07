"""End-to-end verification of the Phase 4 single-image capabilities.

Runs the real API pipeline (upload -> analyze) against synthetic georeferenced rasters and
prints the actual model output: answers, measured scores, evidence kinds, detection boxes,
and geographic coordinates.

    python scripts/verify_phase4.py

Writes ``scripts/phase4_report.txt`` as well as stdout. Nothing here is simulated - every
number comes from a real forward pass. The imagery is synthetic, so the *semantics* of the
answers are not meaningful; what is verified is that the pipeline produces genuine model
output, evidence, confidence, and traces.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

REPORT = Path(__file__).resolve().parent / "phase4_report.txt"


def emit(message: str = "") -> None:
    print(message, flush=True)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_scene() -> bytes:
    """A synthetic georeferenced scene: water strip, vegetation, and building blocks."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    size = 384
    array = np.zeros((3, size, size), dtype="uint8")

    # Vegetation background.
    array[0, :, :] = 70
    array[1, :, :] = 130
    array[2, :, :] = 60

    # A water channel across the upper third.
    array[0, 40:90, :] = 30
    array[1, 40:90, :] = 80
    array[2, 40:90, :] = 190

    # Grey building blocks in the lower half.
    for row in range(230, 350, 40):
        for col in range(40, 340, 50):
            array[:, row : row + 26, col : col + 34] = 185

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
        transform=from_origin(77.5800, 12.9900, 0.0004, 0.0004),
    ) as dst:
        dst.write(array)
    return buffer.getvalue()


def main() -> int:
    REPORT.unlink(missing_ok=True)

    os.environ.setdefault("APP_ENV", "development")
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        health = client.get("/api/v1/health").json()
        emit(f"health: {health['status']} | device={health['device']}")

        models = client.get("/api/v1/models").json()
        by_status: dict[str, list[str]] = {}
        for tool in models["tools"]:
            by_status.setdefault(tool["status"], []).append(tool["id"])
        for status in sorted(by_status):
            emit(f"registry {status}: {', '.join(sorted(by_status[status]))}")
        emit()

        upload = client.post(
            "/api/v1/uploads",
            files=[("file", ("verify_scene.tif", build_scene(), "image/tiff"))],
            data={"role": "optical", "modality": "optical"},
        )
        if upload.status_code != 201:
            emit(f"UPLOAD FAILED {upload.status_code}: {upload.text[:400]}")
            return 1
        asset = upload.json()["asset"]
        emit(
            f"uploaded {asset['original_filename']}: {asset['width']}x{asset['height']}, "
            f"{asset['band_count']} bands, EPSG:{asset['georeference']['crs_epsg']}"
        )
        emit()

        queries = [
            "What land cover dominates this image?",
            "Is water present in this image?",
            "Are there buildings in this scene?",
            "How many buildings are there?",
            "Describe this scene",
            "Highlight the water in this image",
            "Locate the buildings",
        ]

        for query in queries:
            response = client.post(
                "/api/v1/analysis", json={"query": query, "asset_ids": [asset["id"]]}
            )
            body = response.json()
            emit("=" * 78)
            emit(f"QUERY   : {query}")
            emit(f"TASK    : {body['task']}   STATUS: {body['status']}")
            emit(f"ANSWER  : {body['answer']}")
            confidence = body["confidence"]
            emit(f"CONF    : {confidence['level']} (score={confidence['score']})")
            for signal in confidence["signals"]:
                emit(f"          - {signal}")

            if body.get("plan") and body["plan"]["steps"]:
                step = body["plan"]["steps"][-1]
                emit(f"TOOL    : {step['tool_id']} v{step['tool_version']}")

            kinds: dict[str, int] = {}
            for item in body["evidence"]:
                kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
            emit(f"EVIDENCE: {kinds or 'none'}")

            for item in body["evidence"]:
                if item["kind"] == "scores":
                    top = item["detail"]["concepts"][:4]
                    listed = ", ".join(
                        f"{c['key']}={c['probability']:.3f}" for c in top
                    )
                    emit(f"          scores: {listed}")
                if item["kind"] == "box":
                    detail = item["detail"]
                    geo = detail.get("geo") or {}
                    center = geo.get("center_lonlat")
                    emit(
                        f"          box {detail['index']}: score={detail['score']:.3f} "
                        f"px={detail['analyzed_pixel_box']}"
                        + (f" lonlat={center}" if center else "")
                    )
                if item["kind"] == "overlay":
                    image = client.get(item["reference"])
                    is_png = image.content.startswith(b"\x89PNG")
                    emit(
                        f"          overlay: {item['reference']} -> HTTP {image.status_code}, "
                        f"{len(image.content)} bytes, png={is_png}"
                    )
                if item["kind"] == "region":
                    detail = item["detail"]
                    geo = detail.get("geo") or {}
                    emit(
                        f"          region: contrast={detail['tile_contrast']} "
                        f"tiles={detail['tiles_merged']}/{detail['grid']} "
                        f"px={detail['analyzed_pixel_box']} "
                        f"lonlat={geo.get('center_lonlat')}"
                    )

            for warning in body["warnings"]:
                emit(f"WARN    : {warning}")
            for issue in body["issues"]:
                emit(f"ISSUE   : [{issue['code']}] {issue['message']}")
            emit(f"TRACE   : {len(body['trace'])} steps")
            emit()

        history = client.get("/api/v1/history").json()
        emit("=" * 78)
        emit(f"history: {history['count']} analyses")

    emit("VERIFICATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
