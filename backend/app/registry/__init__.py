"""Task and model/tool registries.

The agent may select ONLY from these registries, and only with allowlisted, validated
parameters. Entries carry honest status: in Phase 1 specialists are declared as
``planned`` with truthful metadata; a status flips to ``available`` only when a real
implementation and checkpoint exist.
"""

from .models import (
    MODEL_REGISTRY,
    ParamSpec,
    ToolSpec,
    ToolStatus,
    get_registry_health,
    list_tools,
    tools_for_task,
)
from .tasks import TASK_REGISTRY, Modality, TaskSpec, TaskType, list_tasks

__all__ = [
    "MODEL_REGISTRY",
    "ParamSpec",
    "ToolSpec",
    "ToolStatus",
    "get_registry_health",
    "list_tools",
    "tools_for_task",
    "TASK_REGISTRY",
    "Modality",
    "TaskSpec",
    "TaskType",
    "list_tasks",
]
