"""Render detection boxes onto an image as visual evidence.

The overlay is a *presentation* of real detections: every box drawn corresponds to a
detection returned by the model, with its score labelled. Nothing decorative is added.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .detection_backend import Detection

BOX_COLOR = (255, 92, 0)
BOX_OUTLINE_WIDTH = 3
LABEL_BACKGROUND = (255, 92, 0)
LABEL_TEXT = (255, 255, 255)


def draw_detections(
    image: Image.Image,
    detections: list[Detection],
    destination: str | Path,
) -> tuple[int, int]:
    """Draw ``detections`` on a copy of ``image`` and save it as PNG.

    Boxes are labelled with their score so the evidence carries its own uncertainty.
    Returns the written image's ``(width, height)``.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for index, detection in enumerate(detections, start=1):
        x_min, y_min, x_max, y_max = detection.box
        # Clamp to the canvas so a box that runs off the edge still renders.
        x_min = max(0.0, min(x_min, canvas.width - 1))
        y_min = max(0.0, min(y_min, canvas.height - 1))
        x_max = max(x_min + 1, min(x_max, canvas.width))
        y_max = max(y_min + 1, min(y_max, canvas.height))

        draw.rectangle([x_min, y_min, x_max, y_max], outline=BOX_COLOR, width=BOX_OUTLINE_WIDTH)

        caption = f"{index}. {detection.score:.2f}"
        text_box = draw.textbbox((0, 0), caption)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        # Place the label inside the image even for boxes touching the top edge.
        label_y = y_min - text_height - 4
        if label_y < 0:
            label_y = y_min + 2

        draw.rectangle(
            [x_min, label_y, x_min + text_width + 6, label_y + text_height + 4],
            fill=LABEL_BACKGROUND,
        )
        draw.text((x_min + 3, label_y + 2), caption, fill=LABEL_TEXT)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)
    return canvas.width, canvas.height
