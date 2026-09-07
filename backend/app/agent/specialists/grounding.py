"""Visual grounding: locate a queried feature and return real bounding boxes.

Backed by ``google/owlvit-base-patch32`` open-vocabulary detection. Each returned box comes
from the model with its own score; boxes are drawn onto an overlay image for visual
evidence, and mapped back through the decimation factor and the raster's affine transform
so georeferenced imagery yields real world coordinates.

Honesty rules enforced here:

- **No detections is reported as "could not localize", with confidence "unable to
  determine"** - not as a confident absence. OWL-ViT is trained on natural imagery and is
  weak on diffuse land cover, so a null result is genuinely inconclusive.
- Every box in the overlay corresponds to a real detection, labelled with its score.
"""

from __future__ import annotations

import re
import uuid

from geospatial import box_to_geo
from ml.inference import ModelUnavailableError, clip_backend, detection_backend, find_concepts
from ml.inference.detection_backend import Detection, DetectionResult
from ml.inference.overlay import draw_detections

from ...core.logging import get_logger
from ...services import storage
from ..types import EvidenceItem, SpecialistResult
from .base import Specialist, SpecialistContext
from .vision_support import baseline_warnings, load_asset_image, model_evidence, preview_evidence

logger = get_logger("satquery.grounding")

#: Detections below this score are not reported as findings.
REPORT_THRESHOLD = 0.08
#: Maximum boxes to draw and report.
MAX_REPORTED = 8

OVERLAY_SUBDIR = "overlays"


def _overlay_relative_path(token: str) -> str:
    return f"{storage.EVIDENCE_DIR}/{OVERLAY_SUBDIR}/{token}.png"


#: Words that carry no localizable meaning. A "target" made only of these is not a target,
#: so "outline it for me" is refused rather than searched for as the phrase "it for me".
_FILLER_WORDS = frozenset(
    {
        "it",
        "them",
        "they",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "one",
        "ones",
        "thing",
        "things",
        "area",
        "region",
        "part",
        "image",
        "scene",
        "raster",
        "picture",
        "photo",
        "please",
        "for",
        "me",
        "us",
        "all",
        "any",
        "the",
        "a",
        "an",
        "in",
        "on",
        "of",
        "and",
    }
)


def _clean_target(raw: str) -> str:
    """Strip filler from an extracted target, returning '' when nothing meaningful remains."""
    tokens = [token for token in re.split(r"[^a-z0-9\-]+", raw.lower()) if token]
    meaningful = [token for token in tokens if token not in _FILLER_WORDS]
    return " ".join(meaningful).strip()


def _target_phrases(context: SpecialistContext) -> tuple[str, list[str]]:
    """Decide what to look for. Returns ``(display_target, prompt_phrases)``.

    Prefers a recognised remote-sensing concept (whose prompt phrasing is known to work),
    then falls back to the target extracted during query understanding.
    """
    concepts = find_concepts(context.query)
    if concepts:
        concept = concepts[0]
        return concept.phrase, [concept.phrase]

    if context.target:
        cleaned = _clean_target(context.target)
        if cleaned:
            return cleaned, [cleaned, f"a {cleaned}"]

    return "", []


class GroundingSpecialist(Specialist):
    """Localizes a queried feature with bounding boxes."""

    tool_id = "grounding"

    def run(self, context: SpecialistContext) -> SpecialistResult:
        asset = context.first_asset()
        if asset is None:
            return SpecialistResult(
                answer="No readable image was found for this request.",
                warnings=["The referenced upload could not be loaded."],
            )

        display_target, phrases = _target_phrases(context)
        if not phrases:
            return SpecialistResult(
                answer=(
                    "No target feature was identified in the request. Name what should be "
                    "located, for example 'highlight the water body' or 'locate the buildings'."
                ),
                score=None,
                details={"refused": True, "reason": "no_target"},
                warnings=["Grounding requires an explicit target."],
            )

        try:
            loaded = load_asset_image(asset)
        except Exception as exc:  # noqa: BLE001
            return SpecialistResult(
                answer="The image could not be read for analysis.",
                warnings=[f"Raster read failed: {type(exc).__name__}."],
            )

        threshold = float(context.parameters.get("score_threshold", REPORT_THRESHOLD))

        try:
            result: DetectionResult = detection_backend.detect(
                loaded.image, phrases, threshold=detection_backend.MIN_SCORE
            )
        except ModelUnavailableError:
            raise

        reported = [d for d in result.detections if d.score >= threshold][:MAX_REPORTED]
        warnings = baseline_warnings(asset, uncalibrated=True)
        warnings.append(
            "Detection uses a model trained on natural imagery; it transfers best to "
            "discrete objects (buildings, aircraft, ships) and is weak on diffuse land "
            "cover such as vegetation or farmland."
        )

        evidence: list[EvidenceItem] = [
            model_evidence(
                result.model_id,
                result.device,
                "OWL-ViT open-vocabulary detection; boxes are model output.",
            )
        ]

        # --- no detector boxes: fall back to coarse region localization -------------
        if not reported:
            return self._coarse_fallback(
                context=context,
                asset=asset,
                loaded=loaded,
                display_target=display_target,
                detector_result=result,
                threshold=threshold,
                evidence=evidence,
                warnings=warnings,
            )

        # --- render the overlay -----------------------------------------------------
        transform = None
        metadata = asset.raster_metadata or {}
        if metadata.get("transform"):
            transform = tuple(metadata["transform"])  # type: ignore[arg-type]

        overlay_reference: str | None = None
        token = uuid.uuid4().hex[:16]
        try:
            overlay_path = storage.absolute_path(_overlay_relative_path(token))
            draw_detections(loaded.image, reported, overlay_path)
            overlay_reference = f"/api/v1/evidence/{token}"
        except Exception:  # noqa: BLE001 - overlay is evidence, not the result itself
            logger.warning("Overlay rendering failed for asset %s", asset.id, exc_info=True)
            warnings.append("The visual overlay could not be rendered for this result.")

        if overlay_reference:
            evidence.append(
                EvidenceItem(
                    kind="overlay",
                    label=f"Detected regions for '{display_target}'",
                    asset_id=asset.id,
                    detail={"box_count": len(reported), "image_size": list(result.image_size)},
                    reference=overlay_reference,
                )
            )

        for index, detection in enumerate(reported, start=1):
            geo = box_to_geo(detection.box, transform, asset.crs, loaded.scale)
            detail: dict[str, object] = {
                "index": index,
                "score": round(detection.score, 4),
                "analyzed_pixel_box": [round(v, 1) for v in detection.box],
                "analyzed_image_size": list(result.image_size),
            }
            if geo:
                detail["geo"] = geo
            evidence.append(
                EvidenceItem(
                    kind="box",
                    label=f"Region {index} ({detection.score:.0%})",
                    asset_id=asset.id,
                    detail=detail,
                )
            )

        evidence.extend(preview_evidence(asset))

        best = reported[0]
        answer = (
            f"Located {len(reported)} region(s) matching '{display_target}'. "
            f"The strongest match scores {best.score:.0%}"
        )
        best_geo = box_to_geo(best.box, transform, asset.crs, loaded.scale)
        if best_geo and "center_lonlat" in best_geo:
            lon, lat = best_geo["center_lonlat"]  # type: ignore[misc]
            answer += f" and is centred near {lat:.4f}, {lon:.4f}"
        answer += "."
        if not asset.is_georeferenced:
            answer += " Coordinates are unavailable because the image is not georeferenced."

        return SpecialistResult(
            answer=answer,
            score=float(best.score),
            evidence=evidence,
            details={
                "target": display_target,
                "detections": len(reported),
                "threshold": threshold,
                "model_id": result.model_id,
                "best_score": round(best.score, 4),
                "method": "detector",
            },
            warnings=warnings,
        )

    # --- coarse fallback ----------------------------------------------------------

    def _coarse_fallback(
        self,
        *,
        context: SpecialistContext,
        asset,
        loaded,
        display_target: str,
        detector_result: DetectionResult,
        threshold: float,
        evidence: list[EvidenceItem],
        warnings: list[str],
    ) -> SpecialistResult:
        """Localize by scoring image tiles when the detector returns nothing.

        The detector is trained on natural photographs and often finds no boxes for diffuse
        overhead land cover. Tile scoring with the remote-sensing CLIP model still yields a
        measured region. The result is labelled **coarse** (tile resolution, not an object
        boundary) so it is never mistaken for a detection.
        """
        rejected = detector_result.best
        try:
            localization = clip_backend.localize_by_patches(loaded.image, display_target)
        except ModelUnavailableError:
            localization = None
        except Exception:  # noqa: BLE001 - fallback must not break the request
            logger.warning("Patch localization failed for asset %s", asset.id, exc_info=True)
            localization = None

        if localization is None:
            note = (
                f" The strongest detector candidate scored {rejected.score:.2f}, below the "
                f"{threshold:.2f} reporting threshold."
                if rejected
                else ""
            )
            evidence.extend(preview_evidence(asset))
            return SpecialistResult(
                answer=(
                    f"Could not localize '{display_target}' in this image.{note} "
                    "This is inconclusive rather than evidence of absence."
                ),
                score=None,
                evidence=evidence,
                details={
                    "target": display_target,
                    "detections": 0,
                    "threshold": threshold,
                    "best_rejected_score": round(rejected.score, 4) if rejected else None,
                    "method": "none",
                },
                warnings=warnings,
            )

        transform = None
        metadata = asset.raster_metadata or {}
        if metadata.get("transform"):
            transform = tuple(metadata["transform"])  # type: ignore[arg-type]

        detection = Detection(
            label=display_target, score=localization.contrast, box=localization.box
        )

        overlay_reference: str | None = None
        token = uuid.uuid4().hex[:16]
        try:
            overlay_path = storage.absolute_path(_overlay_relative_path(token))
            draw_detections(loaded.image, [detection], overlay_path)
            overlay_reference = f"/api/v1/evidence/{token}"
        except Exception:  # noqa: BLE001
            logger.warning("Overlay rendering failed for asset %s", asset.id, exc_info=True)
            warnings.append("The visual overlay could not be rendered for this result.")

        geo = box_to_geo(localization.box, transform, asset.crs, loaded.scale)

        if overlay_reference:
            evidence.append(
                EvidenceItem(
                    kind="overlay",
                    label=f"Coarse region for '{display_target}'",
                    asset_id=asset.id,
                    detail={
                        "box_count": 1,
                        "method": "clip_patch_scoring",
                        "grid": localization.grid,
                        "tile_contrast": round(localization.contrast, 4),
                        "image_size": list(detector_result.image_size),
                    },
                    reference=overlay_reference,
                )
            )

        region_detail: dict[str, object] = {
            "index": 1,
            "method": "clip_patch_scoring",
            "grid": f"{localization.grid}x{localization.grid}",
            "tiles_merged": localization.tile_count,
            "tile_contrast": round(localization.contrast, 4),
            "tile_similarity": round(localization.similarity, 4),
            "analyzed_pixel_box": [round(v, 1) for v in localization.box],
            "analyzed_image_size": list(detector_result.image_size),
        }
        if geo:
            region_detail["geo"] = geo
        evidence.append(
            EvidenceItem(
                kind="region",
                label=f"Coarse region for '{display_target}'",
                asset_id=asset.id,
                detail=region_detail,
            )
        )
        evidence.extend(preview_evidence(asset))

        warnings.append(
            "No object boxes were detected, so the region was estimated by scoring a "
            f"{localization.grid}x{localization.grid} tile grid. It marks the area most "
            "consistent with the target, not an object outline."
        )

        answer = (
            f"No distinct object boundary was detected for '{display_target}'. The area most "
            f"consistent with it spans {localization.tile_count} of "
            f"{localization.grid * localization.grid} image tiles "
            f"(tile contrast {localization.contrast:.0%})"
        )
        if geo and "center_lonlat" in geo:
            lon, lat = geo["center_lonlat"]  # type: ignore[misc]
            answer += f", centred near {lat:.4f}, {lon:.4f}"
        answer += "."

        return SpecialistResult(
            answer=answer,
            # Tile-level agreement is weaker evidence than a detector box, so the measured
            # contrast is scaled down: it must not imply detector-grade certainty.
            score=float(localization.contrast * 0.5),
            evidence=evidence,
            details={
                "target": display_target,
                "detections": 0,
                "coarse_region": True,
                "method": "clip_patch_scoring",
                "threshold": threshold,
                "best_rejected_score": round(rejected.score, 4) if rejected else None,
            },
            warnings=warnings,
        )
