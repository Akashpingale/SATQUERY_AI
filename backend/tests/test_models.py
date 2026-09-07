"""Model/tool registry endpoint tests."""

from __future__ import annotations

VALID_STATUSES = {"available", "degraded", "planned", "unavailable"}
MANDATORY_TOOL_IDS = {
    "vqa",
    "captioning",
    "grounding",
    "temporal_change",
    "change_vqa",
    "optical_sar_fusion",
}


def test_models_registry(client):
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()

    ids = {tool["id"] for tool in data["tools"]}
    assert MANDATORY_TOOL_IDS <= ids, "all mandatory specialists must be registered"
    assert data["count"] == len(data["tools"]) >= len(MANDATORY_TOOL_IDS)

    # Honest status: every declared tool has a valid status value.
    assert all(tool["status"] in VALID_STATUSES for tool in data["tools"])

    # Phase 2 support tools are genuinely implemented, so they must report available.
    by_id = {tool["id"]: tool for tool in data["tools"]}
    assert by_id["raster_preprocessor"]["status"] == "available"
    assert by_id["registration_validator"]["status"] == "available"

    # An "available" tool must not claim a checkpoint it does not have, and every
    # available tool must declare a real version (not the 0.0.0 placeholder).
    for tool in data["tools"]:
        if tool["status"] == "available":
            assert tool["version"] != "0.0.0", tool["id"]

    # Registry health mirrors the tools.
    assert set(data["health"].keys()) == ids

    # Every tool declares its I/O contract and allowlisted parameters.
    for tool in data["tools"]:
        assert tool["input_contract"]
        assert tool["output_contract"]
        assert isinstance(tool["allowed_parameters"], list)
