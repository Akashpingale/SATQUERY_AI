"""Phase 8 FE-3: downloadable analysis report.

Reports are generated from the real persisted analysis and downloaded through the API. These
tests upload synthetic rasters, run genuine analyses, then fetch the report and assert it
contains the real stored fields - never a fabricated answer, and always XSS-safe.
"""

from __future__ import annotations

import numpy as np
import pytest

from .raster_fixtures import geotiff_bytes, optical_sar_scene, textured_scene, with_block

TIFF = "image/tiff"
UTM = "EPSG:32643"
GRID = {"crs": UTM, "origin": (500_000.0, 1_500_000.0), "resolution": 10.0, "dtype": "float32"}


def _upload(client, content, name, role, modality="optical"):
    r = client.post(
        "/api/v1/uploads",
        files={"file": (name, content, TIFF)},
        data={"role": role, "modality": modality},
    )
    assert r.status_code == 201, r.text
    return r.json()["asset"]["id"]


def _analyze(client, query, ids):
    r = client.post("/api/v1/analysis", json={"query": query, "asset_ids": ids})
    assert r.status_code == 201, r.text
    return r.json()


def _report(client, analysis_id):
    r = client.get(f"/api/v1/analysis/{analysis_id}/report")
    assert r.status_code == 200, r.text
    return r


# --- basic contract ---------------------------------------------------------------


def test_report_downloads_as_html_attachment(client):
    asset = _upload(client, geotiff_bytes(), "rep_meta.tif", "single", "optical")
    body = _analyze(client, "What CRS and resolution does this image use?", [asset])
    r = _report(client, body["id"])

    assert "text/html" in r.headers["content-type"]
    disposition = r.headers["content-disposition"]
    assert "attachment" in disposition
    assert body["id"] in disposition and disposition.endswith('.html"')
    assert r.text.startswith("<!DOCTYPE html>")


def test_report_contains_the_real_analysis_fields(client):
    asset = _upload(client, geotiff_bytes(), "rep_fields.tif", "single", "optical")
    body = _analyze(client, "What CRS and resolution does this image use?", [asset])
    html = _report(client, body["id"]).text

    assert "SatQuery AI" in html
    assert body["id"] in html
    assert "metadata_inspection" in html
    assert "EPSG:4326" in html  # the real measured answer content
    assert body["confidence"]["level"] in html
    assert "Execution trace" in html
    assert "rep_fields.tif" in html  # input asset metadata
    # The standing, honest caveat is always present.
    assert "not yet benchmark-validated" in html


def test_report_is_xss_safe(client):
    """A hostile query/filename must be escaped, never rendered as live markup."""
    asset = _upload(client, geotiff_bytes(), "rep_xss.tif", "single", "optical")
    body = _analyze(client, "What is this? <script>alert('x')</script>", [asset])
    html = _report(client, body["id"]).text

    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_report_missing_optional_evidence_still_renders(client):
    """A metadata analysis has no rendered overlay - the report must not break."""
    asset = _upload(client, geotiff_bytes(), "rep_noimg.tif", "single", "optical")
    body = _analyze(client, "What bands and data type does this image have?", [asset])
    html = _report(client, body["id"]).text
    assert "Evidence" in html
    assert body["id"] in html


# --- honest handling of non-answers -----------------------------------------------


def test_report_for_unsupported_query_shows_no_fabricated_answer(client):
    asset = _upload(client, geotiff_bytes(), "rep_unsup.tif", "single", "optical")
    body = _analyze(client, "write me a poem about the ocean", [asset])
    assert body["status"] == "rejected"
    html = _report(client, body["id"]).text

    assert "rejected" in html
    assert "No answer was produced" in html
    assert "Issues" in html


def test_report_for_inconclusive_temporal_is_honest(client):
    """A misregistered pair is inconclusive; the report must say so, not invent change."""
    scene = textured_scene(size=96, seed=401)
    shifted = np.roll(scene, shift=2, axis=2)
    t1 = _upload(client, geotiff_bytes(array=scene, **GRID), "rep_mr1.tif", "t1")
    t2 = _upload(client, geotiff_bytes(array=shifted, **GRID), "rep_mr2.tif", "t2")
    body = _analyze(client, "What changed between these dates?", [t1, t2])
    html = _report(client, body["id"]).text
    assert "Unable to determine" in html


# --- per-task evidence embedding --------------------------------------------------


def test_grounding_report_embeds_the_overlay_image(client, models_ready):
    asset = _upload(client, geotiff_bytes(), "rep_ground.tif", "single", "optical")
    body = _analyze(client, "Highlight the water body in this image", [asset])
    html = _report(client, body["id"]).text
    # An overlay or region evidence image should be embedded as a data URI.
    assert "data:image/png;base64," in html


def test_temporal_report_embeds_the_change_map(client):
    scene = textured_scene(size=96, seed=403)
    later = with_block(scene, rows=(20, 55), cols=(25, 65), value=5400.0)
    t1 = _upload(client, geotiff_bytes(array=scene, **GRID), "rep_t1.tif", "t1")
    t2 = _upload(client, geotiff_bytes(array=later, **GRID), "rep_t2.tif", "t2")
    body = _analyze(client, "What changed between these two dates?", [t1, t2])
    html = _report(client, body["id"]).text

    assert body["status"] == "completed"
    assert "change_map" in html
    assert "data:image/png;base64," in html  # the rendered change map is embedded


def test_fusion_report_embeds_cross_modal_evidence(client):
    optical, sar = optical_sar_scene()
    o = _upload(
        client,
        geotiff_bytes(array=optical, band_names=["Red", "Green", "NIR"], **GRID),
        "rep_o.tif",
        "optical",
        "optical",
    )
    s = _upload(client, geotiff_bytes(array=sar, **GRID), "rep_s.tif", "sar", "sar")
    body = _analyze(client, "Analyze these optical and SAR images together.", [o, s])
    html = _report(client, body["id"]).text

    assert body["status"] == "completed"
    assert "fusion_composite" in html
    assert "data:image/png;base64," in html
    assert "complementarity" in html.lower()


def test_adapted_model_report_names_the_adapted_tool(client, models_ready):
    from ml.inference import describe_adapter

    if not describe_adapter().present:
        pytest.skip("Phase 5 adapter not present on disk")

    asset = _upload(client, geotiff_bytes(), "rep_adapt.tif", "single", "optical")
    body = _analyze(client, "What land cover dominates this image?", [asset])
    html = _report(client, body["id"]).text
    # The provenance/trace section records the variant that ran.
    assert "adapted" in html.lower()


# --- 404 -------------------------------------------------------------------------


def test_report_for_unknown_analysis_is_404(client):
    r = client.get("/api/v1/analysis/does-not-exist/report")
    assert r.status_code == 404


# --- unit-level determinism -------------------------------------------------------


@pytest.mark.usefixtures("client")
def test_render_is_deterministic_and_pure():
    """Same record → same report (modulo the generation timestamp line)."""
    from app.reports.html_report import render_analysis_report

    class _Rec:
        id = "abc123"
        status = "completed"
        task = "vqa"
        query = "test query"
        model_id = "vqa"
        model_version = "clip-rsicd-v2"
        confidence = 0.55
        confidence_level = "Medium"
        created_at = None
        input_metadata = {
            "asset_ids": [],
            "answer": "an answer",
            "confidence_signals": ["s1"],
            "warnings": [],
        }
        evidence_refs = {"items": []}
        trace = {"entries": [{"step": "Validation", "status": "ok", "detail": "ok"}]}
        error = None

    first = render_analysis_report(_Rec(), [])
    second = render_analysis_report(_Rec(), [])
    # Only the "Generated" timestamp differs; strip it before comparing.
    import re

    strip = lambda h: re.sub(r"Generated</td><td>[^<]+</td>", "", h)  # noqa: E731
    assert strip(first) == strip(second)
    assert "an answer" in first
    assert "test query" in first
