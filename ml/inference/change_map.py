"""Rendering measured change as inspectable images.

Two artifacts, both derived entirely from the arrays the detector measured:

- :func:`render_change_map` - the change mask and magnitude drawn over the later image.
- :func:`render_comparison_strip` - T1 | T2 | change side by side, which is what makes a
  claim checkable by eye.

Nothing decorative is drawn. A pixel is only tinted when it is in
:attr:`ChangeResult.mask`, which means its measured magnitude cleared both thresholds and it
survived coherence filtering. Region outlines come from the labelled connected components,
so a box always encloses real flagged pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from geospatial.resample import AlignedPair

from .temporal import ChangeResult

#: Tint applied to changed pixels, ramped from low to high magnitude. Amber to red reads as
#: "attention" without implying a semantic class.
LOW_TINT = (255, 214, 102)
HIGH_TINT = (214, 40, 40)
#: Outline colour for reported regions.
OUTLINE = (0, 200, 255)
#: Maximum opacity of the change tint, so the underlying imagery stays visible.
MAX_TINT_ALPHA = 0.80


def _stretch(plane: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Percentile-stretch a float plane to uint8 for display."""
    data = plane.astype("float64", copy=True)
    finite = np.isfinite(data)
    if valid is not None:
        finite &= valid
    pool = data[finite]
    if pool.size == 0:
        return np.zeros(data.shape, dtype="uint8")
    lo, hi = np.percentile(pool, (2.0, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(pool.min()), float(pool.max())
    if hi <= lo:
        return np.zeros(data.shape, dtype="uint8")
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0) * 255.0
    scaled[~finite] = 0
    return scaled.astype("uint8")


def _to_rgb(data: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Render a ``(bands, H, W)`` array as an ``(H, W, 3)`` uint8 image.

    Three or more bands render as RGB from the first three; fewer replicate to grey, which
    is what SAR and single-band products need.
    """
    if data.shape[0] >= 3:
        return np.dstack([_stretch(data[index], valid) for index in range(3)])
    grey = _stretch(data[0], valid)
    return np.stack([grey] * 3, axis=-1)


def _greyscale_base(data: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Desaturated backdrop so the change tint stands out against it."""
    rgb = _to_rgb(data, valid).astype("float32")
    luminance = rgb @ np.array([0.299, 0.587, 0.114], dtype="float32")
    # Lifted and flattened: a mid-grey backdrop keeps tinted pixels legible.
    base = 60.0 + 0.55 * luminance
    return np.clip(np.stack([base] * 3, axis=-1), 0, 255).astype("uint8")


def _tinted(result: ChangeResult, base: np.ndarray) -> np.ndarray:
    """Blend the magnitude-ramped change tint onto ``base`` where the mask is set."""
    canvas = base.astype("float32").copy()
    mask = result.mask
    if not np.any(mask):
        return canvas.astype("uint8")

    magnitudes = result.magnitude[mask]
    # Ramp across the observed range of *flagged* magnitudes, so the colour scale is
    # meaningful for this scene rather than arbitrary.
    lo = float(result.threshold)
    hi = float(magnitudes.max())
    span = max(hi - lo, 1e-6)
    strength = np.clip((result.magnitude - lo) / span, 0.0, 1.0)

    low = np.array(LOW_TINT, dtype="float32")
    high = np.array(HIGH_TINT, dtype="float32")
    ramp = low[None, None, :] + (high - low)[None, None, :] * strength[..., None]

    alpha = (0.45 + 0.35 * strength) * MAX_TINT_ALPHA
    alpha = np.where(mask, alpha, 0.0)[..., None]
    canvas = canvas * (1.0 - alpha) + ramp * alpha
    return np.clip(canvas, 0, 255).astype("uint8")


def _draw_regions(image: Image.Image, result: ChangeResult) -> None:
    """Outline and number the reported regions."""
    draw = ImageDraw.Draw(image)
    for region in result.regions:
        x0, y0, x1, y1 = region.box
        draw.rectangle([x0, y0, x1, y1], outline=OUTLINE, width=2)
        label = str(region.index)
        text_box = draw.textbbox((0, 0), label)
        pad = 3
        chip_w = text_box[2] - text_box[0] + 2 * pad
        chip_h = text_box[3] - text_box[1] + 2 * pad
        chip_x = min(x0, max(0, image.width - chip_w))
        chip_y = max(0, y0 - chip_h)
        draw.rectangle([chip_x, chip_y, chip_x + chip_w, chip_y + chip_h], fill=OUTLINE)
        draw.text((chip_x + pad, chip_y + pad), label, fill=(0, 0, 0))


def render_change_map(
    aligned: AlignedPair,
    result: ChangeResult,
    destination: str | Path,
) -> tuple[int, int]:
    """Write a PNG showing measured change over the later image.

    Returns the ``(width, height)`` of the written image.
    """
    base = _greyscale_base(aligned.second, aligned.valid)
    canvas = _tinted(result, base)
    image = Image.fromarray(canvas, mode="RGB")
    _draw_regions(image, result)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return image.width, image.height


def crop_pair(
    aligned: AlignedPair,
    box: tuple[int, int, int, int],
    *,
    pad: int = 6,
    min_side: int = 24,
) -> tuple[Image.Image, Image.Image] | None:
    """Crop the same region from both dates, for before/after semantic comparison.

    A little padding is included so the region is seen in context rather than as a bare
    rectangle. Returns ``None`` when the region is too small for the crop to carry usable
    content - a 6-pixel patch cannot support a land-cover reading, and pretending otherwise
    would manufacture a semantic claim out of nothing.
    """
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(aligned.width - 1, x1 + pad)
    y1 = min(aligned.height - 1, y1 + pad)
    if (x1 - x0) < min_side or (y1 - y0) < min_side:
        return None

    window = (slice(y0, y1 + 1), slice(x0, x1 + 1))
    valid_window = aligned.valid[window]
    before = _to_rgb(aligned.first[:, window[0], window[1]], valid_window)
    after = _to_rgb(aligned.second[:, window[0], window[1]], valid_window)
    return Image.fromarray(before, mode="RGB"), Image.fromarray(after, mode="RGB")


def render_comparison_strip(
    aligned: AlignedPair,
    result: ChangeResult,
    destination: str | Path,
    *,
    labels: tuple[str, str, str] = ("T1 (earlier)", "T2 (later)", "Measured change"),
    gap: int = 8,
    caption_height: int = 18,
) -> tuple[int, int]:
    """Write a labelled T1 | T2 | change strip on the common grid.

    All three panels share one grid, so a feature at the same position in each panel is the
    same ground location. That is what makes the strip usable as evidence rather than
    decoration.
    """
    panels = [
        _to_rgb(aligned.first, aligned.valid),
        _to_rgb(aligned.second, aligned.valid),
        _tinted(result, _greyscale_base(aligned.second, aligned.valid)),
    ]

    height, width = panels[0].shape[0], panels[0].shape[1]
    total_width = width * 3 + gap * 2
    total_height = height + caption_height

    strip = Image.new("RGB", (total_width, total_height), color=(17, 20, 24))
    for index, panel in enumerate(panels):
        strip.paste(Image.fromarray(panel, mode="RGB"), (index * (width + gap), caption_height))

    # Region outlines belong only on the change panel.
    change_panel = Image.fromarray(panels[2], mode="RGB")
    _draw_regions(change_panel, result)
    strip.paste(change_panel, (2 * (width + gap), caption_height))

    draw = ImageDraw.Draw(strip)
    for index, label in enumerate(labels):
        draw.text((index * (width + gap) + 3, 4), label, fill=(226, 232, 240))

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    strip.save(destination, format="PNG", optimize=True)
    return strip.width, strip.height
