"""Health and root endpoint tests."""

from __future__ import annotations


def test_health_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["environment"] == "test"
    assert data["version"]
    assert data["device"] == "cpu"
    assert data["database"]["connected"] is True
    assert data["database"]["dialect"] == "sqlite"


def test_root_banner(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"]
    assert data["health"] == "/api/v1/health"


def test_request_id_header(client):
    resp = client.get("/api/v1/health")
    assert resp.headers.get("X-Request-ID")
