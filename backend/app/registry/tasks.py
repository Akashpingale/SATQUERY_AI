"""Supported task types and their specifications.

These describe *what* the system can be asked to do and the input shape each task
expects. The agent's query-understanding step (Phase 3) maps a natural-language query
plus the detected input configuration onto one of these tasks.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class Modality(StrEnum):
    """Imagery modalities the system understands.

    ``UNKNOWN`` is a real state, not an error: modality detection is a hint and a user
    may decline to declare one. Tasks never *require* ``UNKNOWN``; it exists so an
    undetermined input can be represented honestly instead of being guessed into a class.
    """

    OPTICAL = "optical"
    SAR = "sar"
    ANY = "any"
    UNKNOWN = "unknown"


class TaskType(StrEnum):
    """Logical analysis tasks."""

    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    TEMPORAL_CHANGE = "temporal_change"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
    METADATA_INSPECTION = "metadata_inspection"
    UNSUPPORTED = "unsupported"


class TaskSpec(BaseModel):
    """Specification for a single task type."""

    type: TaskType
    title: str
    description: str
    min_images: int
    max_images: int
    modalities: list[Modality]
    phase: int  # master-plan phase where this becomes fully available


TASK_REGISTRY: dict[TaskType, TaskSpec] = {
    TaskType.VQA: TaskSpec(
        type=TaskType.VQA,
        title="Visual Question Answering",
        description="Answer a natural-language question about a single image.",
        min_images=1,
        max_images=1,
        modalities=[Modality.OPTICAL, Modality.SAR],
        phase=4,
    ),
    TaskType.CAPTIONING: TaskSpec(
        type=TaskType.CAPTIONING,
        title="Captioning",
        description="Generate a remote-sensing-aware description of a single image.",
        min_images=1,
        max_images=1,
        modalities=[Modality.OPTICAL, Modality.SAR],
        phase=4,
    ),
    TaskType.GROUNDING: TaskSpec(
        type=TaskType.GROUNDING,
        title="Grounding",
        description="Localize a queried object/region with a box and/or mask.",
        min_images=1,
        max_images=1,
        modalities=[Modality.OPTICAL, Modality.SAR],
        phase=4,
    ),
    TaskType.TEMPORAL_CHANGE: TaskSpec(
        type=TaskType.TEMPORAL_CHANGE,
        title="Temporal Change",
        description="Describe changes between two co-located images (T1, T2).",
        min_images=2,
        max_images=2,
        modalities=[Modality.OPTICAL, Modality.SAR],
        phase=6,
    ),
    TaskType.CHANGE_VQA: TaskSpec(
        type=TaskType.CHANGE_VQA,
        title="Change VQA",
        description="Answer a question about the change between two images (T1, T2).",
        min_images=2,
        max_images=2,
        modalities=[Modality.OPTICAL, Modality.SAR],
        phase=6,
    ),
    TaskType.OPTICAL_SAR_FUSION: TaskSpec(
        type=TaskType.OPTICAL_SAR_FUSION,
        title="Optical-SAR Fusion",
        description="Jointly analyze co-registered optical and SAR imagery.",
        min_images=2,
        max_images=2,
        modalities=[Modality.OPTICAL, Modality.SAR],
        phase=7,
    ),
    TaskType.METADATA_INSPECTION: TaskSpec(
        type=TaskType.METADATA_INSPECTION,
        title="Metadata Inspection",
        description="Report raster metadata (CRS, transform, bounds, bands, resolution).",
        min_images=1,
        max_images=2,
        modalities=[Modality.ANY],
        phase=2,
    ),
    TaskType.UNSUPPORTED: TaskSpec(
        type=TaskType.UNSUPPORTED,
        title="Unsupported",
        description="Query could not be mapped to a supported task.",
        min_images=0,
        max_images=2,
        modalities=[Modality.ANY],
        phase=0,
    ),
}


def list_tasks() -> list[TaskSpec]:
    """Return all task specifications."""
    return list(TASK_REGISTRY.values())
