"""Metadata inspection specialists.

These are real, fully implemented capabilities backed by the Phase 2 geospatial layer:

- :class:`RasterPreprocessorSpecialist` reports factual raster metadata for one image.
- :class:`RegistrationValidatorSpecialist` reports whether two images may be analyzed
  together.

Everything they return is measured from the file or computed by ``geospatial`` - there
are no model outputs and nothing is estimated.
"""

from __future__ import annotations

from geospatial import check_pair_compatibility

from ...services.ingestion import metadata_from_asset
from ..types import EvidenceItem, SpecialistResult
from .base import Specialist, SpecialistContext


def _describe_asset(asset, metadata) -> str:
    """One-sentence factual summary of a raster."""
    parts = [
        f"{asset.original_filename} is a {asset.width} x {asset.height} pixel "
        f"{asset.driver or 'raster'} with {asset.band_count} band(s) of type {asset.dtype}"
    ]
    if asset.is_georeferenced and metadata is not None:
        crs = f"EPSG:{asset.crs_epsg}" if asset.crs_epsg else (asset.crs or "an unnamed CRS")
        parts.append(f"georeferenced in {crs}")
        if metadata.resolution:
            unit = "degrees" if asset.crs_epsg == 4326 else "units"
            parts.append(
                f"at a pixel size of {metadata.resolution[0]:g} x "
                f"{metadata.resolution[1]:g} {unit}"
            )
        if metadata.bounds:
            left, bottom, right, top = metadata.bounds
            parts.append(
                f"covering bounds {left:.4f}, {bottom:.4f} to {right:.4f}, {top:.4f}"
            )
    else:
        parts.append(
            "not georeferenced, so coordinates, ground area, and alignment are unavailable"
        )
    return ", ".join(parts) + "."


class RasterPreprocessorSpecialist(Specialist):
    """Reports factual metadata for the supplied image(s)."""

    tool_id = "raster_preprocessor"

    def run(self, context: SpecialistContext) -> SpecialistResult:
        assets = context.assets()
        if not assets:
            return SpecialistResult(
                answer="No readable imagery was found for this request.",
                warnings=["The referenced uploads could not be loaded."],
            )

        sentences: list[str] = []
        evidence: list[EvidenceItem] = []
        warnings: list[str] = []

        for asset in assets:
            metadata = metadata_from_asset(asset)
            sentences.append(_describe_asset(asset, metadata))

            detail: dict[str, object] = {
                "filename": asset.original_filename,
                "width": asset.width,
                "height": asset.height,
                "band_count": asset.band_count,
                "dtype": asset.dtype,
                "driver": asset.driver,
                "is_georeferenced": asset.is_georeferenced,
                "crs": asset.crs,
                "crs_epsg": asset.crs_epsg,
                "modality": asset.modality,
                "detected_modality": asset.detected_modality,
            }
            if metadata is not None:
                detail["resolution"] = list(metadata.resolution) if metadata.resolution else None
                detail["bounds"] = list(metadata.bounds) if metadata.bounds else None
                detail["nodata"] = metadata.nodata

            evidence.append(
                EvidenceItem(
                    kind="metadata",
                    label=f"Metadata for {asset.original_filename}",
                    asset_id=asset.id,
                    detail=detail,
                )
            )

            if asset.preview_path:
                evidence.append(
                    EvidenceItem(
                        kind="preview",
                        label=f"Preview of {asset.original_filename}",
                        asset_id=asset.id,
                        reference=f"/api/v1/uploads/{asset.id}/preview",
                    )
                )

            if not asset.is_georeferenced:
                warnings.append(
                    f"{asset.original_filename} has no CRS or transform, so geospatial "
                    "measurements cannot be derived from it."
                )

        return SpecialistResult(
            answer=" ".join(sentences),
            score=1.0,  # measured directly from the file, not estimated
            evidence=evidence,
            details={"asset_count": len(assets)},
            warnings=warnings,
        )


class RegistrationValidatorSpecialist(Specialist):
    """Reports whether two images can be analyzed together."""

    tool_id = "registration_validator"

    def run(self, context: SpecialistContext) -> SpecialistResult:
        assets = context.assets()
        if len(assets) < 2:
            return SpecialistResult(
                answer="A compatibility check needs two images.",
                warnings=["Fewer than two readable images were supplied."],
            )

        first, second = assets[0], assets[1]
        meta_a = metadata_from_asset(first)
        meta_b = metadata_from_asset(second)
        if meta_a is None or meta_b is None:
            return SpecialistResult(
                answer="Metadata could not be read for one of the images, so compatibility "
                "cannot be verified.",
                warnings=["Re-upload the affected file."],
            )

        result = check_pair_compatibility(
            meta_a,
            meta_b,
            label_a=first.original_filename,
            label_b=second.original_filename,
        )

        if result.compatible:
            summary = (
                f"{first.original_filename} and {second.original_filename} can be analyzed "
                "together"
            )
            if result.overlap_ratio is not None:
                summary += f" with {result.overlap_ratio:.0%} spatial overlap"
            extra = []
            if result.resampling_required:
                extra.append("resampling to a common grid is required")
            if result.alignment_required:
                extra.append("alignment is required")
            summary += f" ({'; '.join(extra)})" if extra else ""
            answer = summary + "."
        else:
            reasons = "; ".join(i.message for i in result.issues if i.severity == "error")
            answer = (
                f"{first.original_filename} and {second.original_filename} cannot be "
                f"analyzed together. {reasons}"
            )

        evidence = [
            EvidenceItem(
                kind="comparison",
                label="Pair compatibility assessment",
                detail={
                    "compatible": result.compatible,
                    "overlap_ratio": result.overlap_ratio,
                    "same_crs": result.same_crs,
                    "resampling_required": result.resampling_required,
                    "alignment_required": result.alignment_required,
                    "issues": [
                        {"code": i.code, "message": i.message, "severity": i.severity}
                        for i in result.issues
                    ],
                    "notes": result.notes,
                },
            )
        ]
        for asset in (first, second):
            if asset.preview_path:
                evidence.append(
                    EvidenceItem(
                        kind="preview",
                        label=f"Preview of {asset.original_filename}",
                        asset_id=asset.id,
                        reference=f"/api/v1/uploads/{asset.id}/preview",
                    )
                )

        return SpecialistResult(
            answer=answer,
            score=1.0 if result.compatible else 0.0,
            evidence=evidence,
            details={
                "compatible": result.compatible,
                "overlap_ratio": result.overlap_ratio,
            },
            warnings=[i.message for i in result.issues if i.severity == "warning"],
        )
