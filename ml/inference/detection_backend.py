"""Open-vocabulary object detection backend for grounding.

Model: ``google/owlvit-base-patch32`` (Apache-2.0). Given a free-text phrase it returns
real bounding boxes with confidence scores, which is what makes visual grounding evidence
possible rather than decorative.

Caveat recorded honestly: OWL-ViT is trained on natural imagery. It transfers reasonably to
overhead views of discrete objects (buildings, aircraft, ships, storage tanks) but is weaker
on diffuse land cover such as farmland or vegetation. Callers surface that limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import (
    OWLVIT_MODEL_ID,
    Availability,
    ModelUnavailableError,
    describe_availability,
    load_cached,
    resolve_device,
)

#: Detections below this score are discarded outright. OWL-ViT scores are low in absolute
#: terms on overhead imagery, so this is deliberately permissive; the caller applies its own
#: threshold on top and always reports the score.
MIN_SCORE = 0.03
MAX_DETECTIONS = 25


@dataclass
class Detection:
    """One detected region, in pixel coordinates of the image that was analyzed."""

    label: str
    score: float
    #: ``(x_min, y_min, x_max, y_max)`` in analyzed-image pixels.
    box: tuple[float, float, float, float]

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)


@dataclass
class DetectionResult:
    """Detections for one image/phrase pair."""

    detections: list[Detection] = field(default_factory=list)
    model_id: str = OWLVIT_MODEL_ID
    device: str = "cpu"
    #: Size of the image the boxes refer to, as ``(width, height)``.
    image_size: tuple[int, int] = (0, 0)

    @property
    def best(self) -> Detection | None:
        return self.detections[0] if self.detections else None


def availability() -> Availability:
    """Whether the detection backend can run right now."""
    return describe_availability(OWLVIT_MODEL_ID)


def _load():
    def factory():
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        device = resolve_device()
        model = OwlViTForObjectDetection.from_pretrained(OWLVIT_MODEL_ID)
        model.eval()
        model.to(device)
        processor = OwlViTProcessor.from_pretrained(OWLVIT_MODEL_ID)
        return model, processor, device

    return load_cached(f"owlvit::{OWLVIT_MODEL_ID}", factory)


def model_version() -> str:
    """A stable version string for provenance."""
    return OWLVIT_MODEL_ID.split("/")[-1]


def detect(image, phrases: list[str], threshold: float = MIN_SCORE) -> DetectionResult:
    """Detect regions matching any of ``phrases``.

    Args:
        image: A PIL RGB image.
        phrases: Free-text descriptions to localize (e.g. ``["a body of water"]``).
        threshold: Minimum score to keep.

    Returns:
        Detections sorted by descending score, capped at :data:`MAX_DETECTIONS`.

    Raises:
        ModelUnavailableError: if the model or its dependencies are missing.
    """
    status = availability()
    if not status.available:
        raise ModelUnavailableError(status.reason)
    if not phrases:
        return DetectionResult(image_size=(image.width, image.height))

    import torch

    model, processor, device = _load()

    with torch.no_grad():
        inputs = processor(text=[phrases], images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]], device=device)
        processed = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

    detections = [
        Detection(
            label=phrases[int(label)] if int(label) < len(phrases) else str(label),
            score=float(score),
            box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
        )
        for score, label, box in zip(
            processed["scores"].tolist(),
            processed["labels"].tolist(),
            processed["boxes"].tolist(),
            strict=True,
        )
    ]
    detections.sort(key=lambda item: item.score, reverse=True)

    return DetectionResult(
        detections=detections[:MAX_DETECTIONS],
        model_id=OWLVIT_MODEL_ID,
        device=device,
        image_size=(image.width, image.height),
    )
