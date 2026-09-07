"""Model/tool registry endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from ....registry.models import get_registry_health, list_tools
from ....schemas.registry import ModelsResponse

router = APIRouter(tags=["registry"])


@router.get("/models", response_model=ModelsResponse, summary="Registered models/tools")
def get_models() -> ModelsResponse:
    tools = list_tools()
    return ModelsResponse(count=len(tools), tools=tools, health=get_registry_health())
