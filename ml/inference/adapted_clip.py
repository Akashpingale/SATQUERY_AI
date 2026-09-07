"""Serving the Phase 5 adapted CLIP alongside the Phase 4 baseline.

Two variants are exposed and kept strictly distinguishable:

- ``baseline`` - the stock Phase 4 model (``flax-community/clip-rsicd-v2``).
- ``adapted``  - that model plus the Phase 5 LoRA adapter trained on BigEarthNet.txt.

The adapted variant is only reported available when a real adapter directory with weight
files exists and loads. If it does not, callers fall back to the baseline and the trace
records that the fallback happened, so a downgrade is never silent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .base import (
    CLIP_RS_MODEL_ID,
    Availability,
    ModelUnavailableError,
    describe_availability,
    load_cached,
    resolve_device,
)


class ClipVariant(StrEnum):
    BASELINE = "baseline"
    ADAPTED = "adapted"


#: Adapter location, overridable so a different adapter can be served without code changes.
DEFAULT_ADAPTER_DIR = "models/phase5/adapter"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def adapter_dir() -> Path:
    """Resolve the adapter directory (absolute)."""
    configured = os.environ.get("PHASE5_ADAPTER_DIR", DEFAULT_ADAPTER_DIR)
    path = Path(configured)
    return path if path.is_absolute() else _repo_root() / path


@dataclass(frozen=True)
class AdapterInfo:
    """Provenance for a loaded adapter, surfaced in traces and evidence."""

    present: bool
    path: str
    base_model: str | None = None
    training_dataset: str | None = None
    training_method: str | None = None
    trainable_parameters: int | None = None
    rank: int | None = None
    alpha: int | None = None
    target_modules: list[str] | None = None
    size_bytes: int | None = None
    reason: str = ""


def describe_adapter() -> AdapterInfo:
    """Inspect the adapter on disk without loading the model.

    Reads the PEFT config that ``save_pretrained`` wrote, so the reported rank/alpha/targets
    are the values actually used for training rather than assumed defaults.
    """
    path = adapter_dir()
    if not path.is_dir():
        return AdapterInfo(present=False, path=str(path), reason="adapter directory not found")

    weight_files = list(path.glob("adapter_model*"))
    if not weight_files:
        return AdapterInfo(
            present=False, path=str(path), reason="no adapter weight file in directory"
        )

    config_path = path / "adapter_config.json"
    base_model = None
    rank = alpha = None
    targets = None
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            base_model = config.get("base_model_name_or_path")
            rank = config.get("r")
            alpha = config.get("lora_alpha")
            modules = config.get("target_modules")
            targets = sorted(modules) if isinstance(modules, (list, set)) else None
        except (OSError, json.JSONDecodeError):
            pass

    size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    return AdapterInfo(
        present=True,
        path=str(path),
        base_model=base_model or CLIP_RS_MODEL_ID,
        training_dataset="BigEarthNet.txt (Lithuania/Summer captioning subset)",
        training_method="LoRA, symmetric contrastive (InfoNCE) objective",
        rank=rank,
        alpha=alpha,
        target_modules=targets,
        size_bytes=size,
    )


def availability(variant: ClipVariant | str = ClipVariant.BASELINE) -> Availability:
    """Whether a variant can run right now."""
    base_status = describe_availability(CLIP_RS_MODEL_ID)
    if not base_status.available:
        return base_status

    if ClipVariant(variant) is ClipVariant.BASELINE:
        return Availability(True)

    info = describe_adapter()
    if not info.present:
        return Availability(False, f"Phase 5 adapter unavailable: {info.reason}.")
    return Availability(True)


def load_variant(variant: ClipVariant | str = ClipVariant.BASELINE):
    """Load and cache ``(model, processor, device)`` for a variant.

    Raises:
        ModelUnavailableError: if the variant cannot be loaded.
    """
    variant = ClipVariant(variant)
    status = availability(variant)
    if not status.available:
        raise ModelUnavailableError(status.reason)

    def factory():
        device = resolve_device()
        if variant is ClipVariant.BASELINE:
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(CLIP_RS_MODEL_ID)
            model.eval()
            model.to(device)
            processor = CLIPProcessor.from_pretrained(CLIP_RS_MODEL_ID)
            return model, processor, device

        # Imported here so the serving path does not require peft unless an adapter is used.
        from ml.training.lora_clip import load_adapted_clip

        model, processor = load_adapted_clip(adapter_dir(), device=device)
        return model, processor, device

    return load_cached(f"clip-variant::{variant}", factory)


def resolve_variant(prefer_adapted: bool = True) -> tuple[ClipVariant, str]:
    """Pick the variant to use, reporting why.

    Returns ``(variant, note)``. The note is written into the execution trace so a fallback
    from adapted to baseline is always visible.
    """
    if not prefer_adapted:
        return ClipVariant.BASELINE, "baseline requested"

    status = availability(ClipVariant.ADAPTED)
    if status.available:
        return ClipVariant.ADAPTED, "Phase 5 adapted model selected"
    return ClipVariant.BASELINE, f"fell back to baseline ({status.reason})"


def variant_version(variant: ClipVariant | str) -> str:
    """A version string identifying exactly what ran."""
    variant = ClipVariant(variant)
    if variant is ClipVariant.BASELINE:
        return CLIP_RS_MODEL_ID.split("/")[-1]
    info = describe_adapter()
    suffix = f"r{info.rank}a{info.alpha}" if info.rank is not None else "unknown"
    return f"{CLIP_RS_MODEL_ID.split('/')[-1]}+lora-ben-txt-{suffix}"
