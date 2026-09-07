"""Shared helpers for the vision specialists.

Keeps image loading, provenance evidence, and the standing model caveats in one place so
the VQA, captioning, and grounding specialists stay focused on their own logic and report
their limitations consistently.
"""

from __future__ import annotations

from dataclasses import dataclass

from geospatial import load_display_image

from ...models_db.raster_asset import RasterAsset
from ...services import storage
from ..types import EvidenceItem

#: Longest side of the image handed to the models. Vision transformers resize to 224px
#: internally, so a larger read wastes time without improving accuracy; 512 keeps enough
#: detail for detection boxes to be meaningful when mapped back to source pixels.
MODEL_INPUT_MAX_SIDE = 512


@dataclass
class LoadedImage:
    """A raster prepared for model input."""

    image: object  # PIL.Image.Image
    #: Decimation factor applied; divide an image coordinate by this for source pixels.
    scale: float
    asset: RasterAsset


def load_asset_image(asset: RasterAsset, max_side: int = MODEL_INPUT_MAX_SIDE) -> LoadedImage:
    """Load an asset as an RGB image suitable for model input.

    Reads are decimated, so this is safe for large rasters. Single-band SAR is replicated
    to three channels by ``load_display_image`` rather than being rejected.
    """
    path = storage.absolute_path(asset.storage_path)
    image, scale = load_display_image(path, max_side=max_side)
    return LoadedImage(image=image, scale=scale, asset=asset)


def preview_evidence(asset: RasterAsset) -> list[EvidenceItem]:
    """Evidence item pointing at the asset's stored preview, when one exists."""
    if not asset.preview_path:
        return []
    return [
        EvidenceItem(
            kind="preview",
            label=f"Preview of {asset.original_filename}",
            asset_id=asset.id,
            reference=f"/api/v1/uploads/{asset.id}/preview",
        )
    ]


def model_evidence(model_id: str, device: str, note: str) -> EvidenceItem:
    """Provenance evidence recording exactly which model produced a result."""
    return EvidenceItem(
        kind="model",
        label="Model provenance",
        detail={"model_id": model_id, "device": device, "note": note},
    )


def variant_evidence(variant: str, note: str) -> EvidenceItem:
    """Evidence recording which model variant ran and why.

    This is what lets a reader tell a baseline answer from an adapted one, and see when a
    requested adapted model fell back to the baseline.
    """
    from ml.inference import describe_adapter, variant_version

    detail: dict[str, object] = {
        "variant": variant,
        "version": variant_version(variant),
        "selection": note,
    }
    if variant == "adapted":
        info = describe_adapter()
        detail.update(
            {
                "base_model": info.base_model,
                "adapter_path": info.path,
                "training_dataset": info.training_dataset,
                "training_method": info.training_method,
                "lora_rank": info.rank,
                "lora_alpha": info.alpha,
            }
        )
    return EvidenceItem(kind="model", label="Model variant", detail=detail)


def adaptation_warnings(variant: str) -> list[str]:
    """Caveats specific to the model variant that ran."""
    if variant != "adapted":
        return []
    return [
        "This answer used the Phase 5 adapted model (LoRA on BigEarthNet.txt captions). "
        "Adaptation was trained on a Lithuania/Summer Sentinel-2 subset, so its benefit on "
        "other regions, seasons, or sensors is not established."
    ]


def baseline_warnings(asset: RasterAsset, *, uncalibrated: bool = True) -> list[str]:
    """Standing caveats that apply to every Phase 4 baseline result.

    These are stated on every result rather than buried in documentation, because a user
    reading an answer needs to know how much weight it carries.
    """
    warnings: list[str] = []
    if uncalibrated:
        warnings.append(
            "Scores come from an uncalibrated zero-shot model and are relative to its "
            "concept vocabulary; treat them as indicative rather than absolute."
        )
    if not asset.is_georeferenced:
        warnings.append(
            f"{asset.original_filename} has no CRS or transform, so results cannot be "
            "tied to ground coordinates."
        )
    return warnings
