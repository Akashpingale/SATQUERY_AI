"""Supported tasks endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from ....registry.tasks import list_tasks
from ....schemas.registry import TasksResponse

router = APIRouter(tags=["registry"])


@router.get("/tasks", response_model=TasksResponse, summary="Supported task types")
def get_tasks() -> TasksResponse:
    tasks = list_tasks()
    return TasksResponse(count=len(tasks), tasks=tasks)
