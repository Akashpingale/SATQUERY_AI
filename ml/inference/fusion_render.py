"""Rendering optical/SAR cross-modal evidence as inspectable images.

Every artifact is derived from the measured, co-registered feature planes, so what the user
sees is the same data the answer is computed from. Nothing decorative is drawn.

Artifacts:
- :func:`render_aligned_optical` / :func:`render_aligned_sar` - the two modalities on the
  identical shared grid, so a feature at a position is the same ground location in both.
- :func:`render_fusion_composite` - a false-colour optical+SAR composite (SAR in red, optical
  brightness in green/blue). Genuinely fused: each channel is a real normalised plane, and a
  colour tells you which modality is strong there.
- :func:`render_complementary_map` - highlights the coherent regions where one modality carries
  information the other does not (SAR-only structure, cross-modal water agreement).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from geospatial.resample import AlignedModalities

from .fusion import (
    SAR_DARK_PERCENTILE,
    STRETCH_PERCENTILES,
    STRUCTURE_TEXTURE,
    CrossModalResult,
    _local_texture,
    _optical_brightness,
    _robust_normalize,
    _sar_intensity,
)

#: Region outline colours by kind.
_KIND_COLOUR = {
    "sar_structure": (0, 200, 255),
    "water_agreement": (80, 160, 255),
    "optical_spectral": (120, 255, 120),
}


def _stretch(plane: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    data = plane.astype("float64", copy=True)
    finite = np.isfinite(data)
    if valid is not None:
        finite &= valid
    pool = data[finite]
    if pool.size == 0:
        return np.zeros(data.shape, dtype="uint8")
    lo, hi = np.percentile(pool, STRETCH_PERCENTILES)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(pool.min()), float(pool.max())
    if hi <= lo:
        return np.zeros(data.shape, dtype="uint8")
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0) * 255.0
    scaled[~finite] = 0
    return scaled.astype("uint8")


def optical_rgb(aligned: AlignedModalities) -> np.ndarray:
    """``(H, W, 3)`` uint8 render of the aligned optical image."""
    if aligned.optical.shape[0] >= 3:
        return np.dstack([_stretch(aligned.optical[i], aligned.valid) for i in range(3)])
    grey = _stretch(aligned.optical[0], aligned.valid)
    return np.stack([grey] * 3, axis=-1)


def sar_grey(aligned: AlignedModalities) -> np.ndarray:
    """``(H, W, 3)`` grey render of the aligned SAR backscatter (log-scaled)."""
    plane, _ = _sar_intensity(aligned, calibrated=False)
    grey = _stretch(plane, aligned.valid)
    return np.stack([grey] * 3, axis=-1)


def optical_pil(aligned: AlignedModalities) -> Image.Image:
    """Aligned optical as a PIL RGB image (for an optional zero-shot land-cover reading)."""
    return Image.fromarray(optical_rgb(aligned), mode="RGB")


def _save(image: Image.Image, destination: str | Path) -> tuple[int, int]:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return image.width, image.height


def render_aligned_optical(aligned: AlignedModalities, destination: str | Path) -> tuple[int, int]:
    return _save(Image.fromarray(optical_rgb(aligned), mode="RGB"), destination)


def render_aligned_sar(aligned: AlignedModalities, destination: str | Path) -> tuple[int, int]:
    return _save(Image.fromarray(sar_grey(aligned), mode="RGB"), destination)


def render_fusion_composite(
    aligned: AlignedModalities, destination: str | Path
) -> tuple[int, int]:
    """False-colour optical+SAR composite: R = SAR backscatter, G/B = optical brightness.

    A standard way to view the two modalities at once. Reddish areas are SAR-bright (rough /
    structured), cyan areas are optically bright with low SAR return (smooth surfaces), and
    grey areas agree. Each channel is a real normalised measurement.
    """
    sar_plane, _ = _sar_intensity(aligned, calibrated=False)
    sar_norm = _robust_normalize(sar_plane, aligned.valid)
    optical = _robust_normalize(_optical_brightness(aligned), aligned.valid)

    red = (sar_norm * 255).astype("uint8")
    green = (optical * 255).astype("uint8")
    blue = (optical * 255).astype("uint8")
    composite = np.dstack([red, green, blue])
    composite[~aligned.valid] = 0
    return _save(Image.fromarray(composite, mode="RGB"), destination)


def render_complementary_map(
    aligned: AlignedModalities, result: CrossModalResult, destination: str | Path
) -> tuple[int, int]:
    """Grey optical backdrop with the measured complementary regions tinted and outlined.

    Tint is applied only to pixels that actually satisfy the measured condition (SAR-only
    structure, or cross-modal water agreement), so no region is decorative.
    """
    base = optical_rgb(aligned).astype("float32")
    luminance = base @ np.array([0.299, 0.587, 0.114], dtype="float32")
    canvas = np.clip(np.stack([60 + 0.55 * luminance] * 3, axis=-1), 0, 255)

    valid = aligned.valid
    sar_plane, _ = _sar_intensity(aligned, calibrated=False)
    sar_norm = _robust_normalize(sar_plane, valid)
    optical_norm = _robust_normalize(_optical_brightness(aligned), valid)
    sar_texture = _local_texture(sar_norm)
    optical_texture = _local_texture(optical_norm)

    sar_only = (sar_texture > STRUCTURE_TEXTURE) & (optical_texture <= STRUCTURE_TEXTURE) & valid
    _tint(canvas, sar_only, _KIND_COLOUR["sar_structure"], 0.55)

    if result.water.available:
        dark = sar_norm <= np.percentile(sar_norm[valid], SAR_DARK_PERCENTILE)
        # water agreement pixels are highlighted only where both modalities concur
        agree = dark & valid
        _tint(canvas, agree, _KIND_COLOUR["water_agreement"], 0.35)

    image = Image.fromarray(np.clip(canvas, 0, 255).astype("uint8"), mode="RGB")
    draw = ImageDraw.Draw(image)
    for region in result.regions:
        colour = _KIND_COLOUR.get(region.kind, (255, 255, 255))
        x0, y0, x1, y1 = region.box
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=2)
        draw.text((min(x0, image.width - 10), max(0, y0 - 12)), str(region.index), fill=colour)
    return _save(image, destination)


def _tint(canvas: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    tint = np.array(colour, dtype="float32")
    canvas[mask] = canvas[mask] * (1.0 - alpha) + tint * alpha
