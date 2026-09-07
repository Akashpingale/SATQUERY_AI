"""Integration tests for the uploads API, including negative cases."""

from __future__ import annotations

from .raster_fixtures import geotiff_bytes, png_bytes

TIFF_TYPE = "image/tiff"


def _upload(client, content: bytes, filename: str = "scene.tif", **form):
    data = {"role": "single", "modality": "unknown", **form}
    return client.post(
        "/api/v1/uploads",
        files={"file": (filename, content, TIFF_TYPE)},
        data=data,
    )


# --- happy path -----------------------------------------------------------------


def test_upload_geotiff_returns_real_metadata(client):
    resp = _upload(client, geotiff_bytes(width=96, height=48, bands=3), modality="optical")
    assert resp.status_code == 201, resp.text

    asset = resp.json()["asset"]
    assert asset["width"] == 96
    assert asset["height"] == 48
    assert asset["band_count"] == 3
    assert asset["dtype"] == "uint8"
    assert asset["driver"] == "GTiff"
    assert asset["modality"] == "optical"
    assert asset["georeference"]["is_georeferenced"] is True
    assert asset["georeference"]["crs_epsg"] == 4326
    assert asset["georeference"]["bounds"] is not None
    assert len(asset["bands"]) == 3
    assert asset["preview_url"].endswith(f"/uploads/{asset['id']}/preview")


def test_preview_is_served_as_png(client):
    asset = _upload(client, geotiff_bytes()).json()["asset"]
    resp = client.get(f"/api/v1/uploads/{asset['id']}/preview")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")


def test_get_and_list_uploads(client):
    asset = _upload(client, geotiff_bytes()).json()["asset"]

    single = client.get(f"/api/v1/uploads/{asset['id']}")
    assert single.status_code == 200
    assert single.json()["id"] == asset["id"]

    listing = client.get("/api/v1/uploads")
    assert listing.status_code == 200
    body = listing.json()
    assert body["count"] >= 1
    assert asset["id"] in [a["id"] for a in body["assets"]]


def test_sar_upload_detects_single_band_float(client):
    resp = _upload(
        client, geotiff_bytes(bands=1, dtype="float32"), filename="sar.tif", modality="sar"
    )
    asset = resp.json()["asset"]

    assert asset["band_count"] == 1
    assert asset["dtype"] == "float32"
    assert asset["modality"] == "sar"
    assert asset["detected_modality"] == "sar"


def test_plain_png_accepted_with_georeferencing_warning(client):
    resp = client.post(
        "/api/v1/uploads",
        files={"file": ("bench.png", png_bytes(), "image/png")},
        data={"role": "single", "modality": "optical"},
    )
    assert resp.status_code == 201

    asset = resp.json()["asset"]
    assert asset["georeference"]["is_georeferenced"] is False
    assert "not_georeferenced" in [w["code"] for w in asset["warnings"]]


def test_delete_upload(client):
    asset = _upload(client, geotiff_bytes()).json()["asset"]

    assert client.delete(f"/api/v1/uploads/{asset['id']}").status_code == 204
    assert client.get(f"/api/v1/uploads/{asset['id']}").status_code == 404


# --- negative cases -------------------------------------------------------------


def test_corrupt_tiff_rejected_with_explanation(client):
    resp = _upload(client, b"II*\x00 not a real tiff " + b"\x00" * 32, filename="bad.tif")

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "unreadable_raster" in [i["code"] for i in detail["issues"]]


def test_unsupported_extension_rejected(client):
    resp = client.post(
        "/api/v1/uploads",
        files={"file": ("payload.exe", b"MZ\x90\x00binary", "application/octet-stream")},
        data={"role": "single", "modality": "unknown"},
    )
    assert resp.status_code == 422
    assert "unsupported_format" in [i["code"] for i in resp.json()["detail"]["issues"]]


def test_disallowed_content_type_rejected(client):
    resp = client.post(
        "/api/v1/uploads",
        files={"file": ("page.tif", geotiff_bytes(), "text/html")},
        data={"role": "single", "modality": "unknown"},
    )
    assert resp.status_code == 415


def test_empty_file_rejected(client):
    resp = _upload(client, b"", filename="empty.tif")
    assert resp.status_code == 422


def test_invalid_role_rejected(client):
    resp = _upload(client, geotiff_bytes(), role="root")
    assert resp.status_code == 422


def test_invalid_modality_rejected(client):
    resp = _upload(client, geotiff_bytes(), modality="infrared")
    assert resp.status_code == 422


def test_malicious_filename_is_sanitized(client):
    """A traversal filename must not escape storage and must be stored safely."""
    resp = _upload(client, geotiff_bytes(), filename="../../../../etc/evil.tif")
    assert resp.status_code == 201
    assert resp.json()["asset"]["original_filename"] == "../../../../etc/evil.tif"

    # The asset is retrievable, proving it was stored under a safe derived name.
    asset_id = resp.json()["asset"]["id"]
    assert client.get(f"/api/v1/uploads/{asset_id}").status_code == 200


def test_unknown_asset_returns_404(client):
    assert client.get("/api/v1/uploads/does-not-exist").status_code == 404
    assert client.get("/api/v1/uploads/does-not-exist/preview").status_code == 404


# --- pair compatibility ---------------------------------------------------------


def test_compatibility_accepts_aligned_pair(client):
    a = _upload(client, geotiff_bytes(), filename="t1.tif", role="t1").json()["asset"]
    b = _upload(client, geotiff_bytes(), filename="t2.tif", role="t2").json()["asset"]

    resp = client.post(
        "/api/v1/uploads/compatibility",
        json={"asset_id_a": a["id"], "asset_id_b": b["id"], "purpose": "temporal"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["compatible"] is True
    assert body["overlap_ratio"] == 1.0
    assert body["same_crs"] is True


def test_compatibility_rejects_non_overlapping_pair(client):
    """The guard that stops change analysis on unrelated images."""
    a = _upload(client, geotiff_bytes(), filename="a.tif", role="t1").json()["asset"]
    b = _upload(
        client, geotiff_bytes(origin=(10.0, 50.0)), filename="b.tif", role="t2"
    ).json()["asset"]

    resp = client.post(
        "/api/v1/uploads/compatibility",
        json={"asset_id_a": a["id"], "asset_id_b": b["id"], "purpose": "temporal"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["compatible"] is False
    assert "no_overlap" in [i["code"] for i in body["issues"]]


def test_compatibility_requires_both_modalities_for_fusion(client):
    a = _upload(client, geotiff_bytes(), filename="o1.tif", modality="optical").json()["asset"]
    b = _upload(client, geotiff_bytes(), filename="o2.tif", modality="optical").json()["asset"]

    resp = client.post(
        "/api/v1/uploads/compatibility",
        json={"asset_id_a": a["id"], "asset_id_b": b["id"], "purpose": "optical_sar"},
    )
    body = resp.json()
    assert body["compatible"] is False
    assert "modality_mismatch" in [i["code"] for i in body["issues"]]


def test_compatibility_accepts_optical_sar_pair(client):
    optical = _upload(client, geotiff_bytes(), filename="o.tif", modality="optical").json()["asset"]
    sar = _upload(
        client, geotiff_bytes(bands=1, dtype="float32"), filename="s.tif", modality="sar"
    ).json()["asset"]

    resp = client.post(
        "/api/v1/uploads/compatibility",
        json={"asset_id_a": optical["id"], "asset_id_b": sar["id"], "purpose": "optical_sar"},
    )
    body = resp.json()
    assert body["compatible"] is True


def test_compatibility_rejects_same_asset_twice(client):
    a = _upload(client, geotiff_bytes()).json()["asset"]
    resp = client.post(
        "/api/v1/uploads/compatibility",
        json={"asset_id_a": a["id"], "asset_id_b": a["id"], "purpose": "temporal"},
    )
    assert resp.status_code == 422


def test_compatibility_unknown_asset_returns_404(client):
    a = _upload(client, geotiff_bytes()).json()["asset"]
    resp = client.post(
        "/api/v1/uploads/compatibility",
        json={"asset_id_a": a["id"], "asset_id_b": "missing", "purpose": "temporal"},
    )
    assert resp.status_code == 404
