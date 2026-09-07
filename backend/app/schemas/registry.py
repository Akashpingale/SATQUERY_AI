"""Response schemas for the model/tool and task discovery endpoints."""

from __future__ import annotations

from pydantic import BaseModel

from ..registry.models import ToolSpec
from ..registry.tasks import TaskSpec


class ModelsResponse(BaseModel):
    count: int
    tools: list[ToolSpec]
    health: dict[str, str]


class TasksResponse(BaseModel):
    count: int
    tasks: list[TaskSpec]
