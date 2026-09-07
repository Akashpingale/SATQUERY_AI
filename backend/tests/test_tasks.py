"""Task registry endpoint tests."""

from __future__ import annotations

MANDATORY_TASKS = {
    "vqa",
    "captioning",
    "grounding",
    "temporal_change",
    "change_vqa",
    "optical_sar_fusion",
}


def test_tasks_registry(client):
    resp = client.get("/api/v1/tasks")
    assert resp.status_code == 200
    data = resp.json()

    types = {task["type"] for task in data["tasks"]}
    assert MANDATORY_TASKS <= types
    assert data["count"] == len(data["tasks"])


def test_temporal_tasks_require_two_images(client):
    resp = client.get("/api/v1/tasks")
    tasks = {task["type"]: task for task in resp.json()["tasks"]}
    for two_image_task in ("temporal_change", "change_vqa", "optical_sar_fusion"):
        assert tasks[two_image_task]["min_images"] == 2
        assert tasks[two_image_task]["max_images"] == 2


def test_single_image_tasks_require_one_image(client):
    resp = client.get("/api/v1/tasks")
    tasks = {task["type"]: task for task in resp.json()["tasks"]}
    for single in ("vqa", "captioning", "grounding"):
        assert tasks[single]["min_images"] == 1
        assert tasks[single]["max_images"] == 1
