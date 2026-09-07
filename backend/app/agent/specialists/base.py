"""The uniform specialist contract.

Every analysis capability - metadata inspection today, VQA/grounding/temporal/fusion in
later phases - implements this one interface. That is what lets the orchestrator treat
models as interchangeable and lets the registry describe them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from ...models_db.raster_asset import RasterAsset
from ..types import InputConfiguration, SpecialistResult


@dataclass
class SpecialistContext:
    """Everything a specialist may read while executing.

    Deliberately narrow: a specialist gets the query, the resolved inputs, validated
    parameters, and database access for asset lookups. It never receives raw request
    data, secrets, or the ability to choose its own tools.
    """

    query: str
    inputs: InputConfiguration
    parameters: dict[str, Any]
    db: Session
    #: Target phrase extracted during query understanding (used by grounding).
    target: str | None = None

    def assets(self) -> list[RasterAsset]:
        """Load the ORM rows for the input images, preserving input order."""
        rows: list[RasterAsset] = []
        for image in self.inputs.images:
            asset = self.db.get(RasterAsset, image.asset_id)
            if asset is not None:
                rows.append(asset)
        return rows

    def first_asset(self) -> RasterAsset | None:
        """The single asset for single-image tasks."""
        assets = self.assets()
        return assets[0] if assets else None


class Specialist(ABC):
    """Base class for all analysis specialists."""

    #: Registry id this specialist implements. Must exist in ``MODEL_REGISTRY``.
    tool_id: str

    @abstractmethod
    def run(self, context: SpecialistContext) -> SpecialistResult:
        """Execute the capability and return the uniform result contract."""
        raise NotImplementedError


class SpecialistUnavailable(RuntimeError):
    """Raised when a registered tool has no runnable implementation.

    This is how the system stays honest: a task that cannot really be executed reports
    unavailability instead of returning a fabricated answer.
    """

    def __init__(self, tool_id: str, message: str | None = None) -> None:
        self.tool_id = tool_id
        super().__init__(
            message
            or f"The '{tool_id}' specialist is registered but has no implementation yet."
        )
