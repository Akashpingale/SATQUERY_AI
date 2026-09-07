"""Zero-shot scoring backend built on CLIP fine-tuned for remote sensing.

Model: ``flax-community/clip-rsicd-v2`` - CLIP fine-tuned on RSICD, a remote-sensing image
captioning dataset. It is therefore genuinely remote-sensing-adapted rather than a purely
natural-image model, but it is still a **baseline**: it is contrastive (not generative), its
softmax scores are uncalibrated, and it has no counting or spatial-reasoning mechanism.
Those limits are surfaced to callers instead of being hidden.

All scores returned here come from real forward passes. Nothing is simulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import (
    CLIP_RS_MODEL_ID,
    Availability,
    ModelUnavailableError,
    load_cached,
)
from .vocabulary import (
    CONCEPTS,
    LAND_COVER_CONCEPTS,
    Concept,
    build_prompts,
    label_for,
)

#: Softmax probability over the full concept vocabulary above which a concept is treated
#: as present. Chosen conservatively; the raw score is always reported alongside the
#: verdict so a reader can apply their own judgement.
PRESENCE_THRESHOLD = 0.10


@dataclass
class ConceptScore:
    """A concept and its measured probability."""

    key: str
    label: str
    probability: float


@dataclass
class ScoreResult:
    """Ranked concept scores from one forward pass."""

    scores: list[ConceptScore] = field(default_factory=list)
    model_id: str = CLIP_RS_MODEL_ID
    device: str = "cpu"
    #: "baseline" or "adapted" - which model produced these scores.
    variant: str = "baseline"
    #: Version string identifying exactly what ran.
    version: str = ""

    @property
    def top(self) -> ConceptScore | None:
        return self.scores[0] if self.scores else None

    def by_key(self, key: str) -> ConceptScore | None:
        for score in self.scores:
            if score.key == key:
                return score
        return None

    def above(self, threshold: float) -> list[ConceptScore]:
        return [score for score in self.scores if score.probability >= threshold]


def availability(variant: str = "baseline") -> Availability:
    """Whether the CLIP backend can run right now for a given variant."""
    from .adapted_clip import availability as variant_availability

    return variant_availability(variant)


def _load(variant: str = "baseline"):
    """Load and cache the model plus processor for a variant."""
    from .adapted_clip import load_variant

    return load_variant(variant)


def _base_module(model):
    """Return the underlying CLIP model, unwrapping PEFT if present.

    ``get_image_features`` / ``get_text_features`` live on the CLIP model; a PeftModel
    proxies ``forward`` but these helpers are reached through the base module. The LoRA
    layers are injected into the base module's submodules, so calling them through the base
    still applies the adapter.
    """
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def model_version(variant: str = "baseline") -> str:
    """A stable version string for provenance in traces and reports."""
    from .adapted_clip import variant_version

    return variant_version(variant)


def _concept_text_embeddings(concepts: tuple[Concept, ...], variant: str = "baseline"):
    """Return normalised, template-averaged text embeddings for ``concepts``.

    Text embeddings depend only on the vocabulary, never on the image, so they are
    computed once per (concept set, variant) and cached. This removes the dominant cost from
    every request: encoding ~70 prompts took several seconds, while encoding one image takes
    a fraction of that. The variant is part of the key because the adapted text tower
    produces different embeddings.
    """
    cache_key = f"clip-text::{variant}::{','.join(c.key for c in concepts)}"

    def factory():
        import torch

        model, processor, device = _load(variant)
        base = _base_module(model)
        prompts: list[str] = []
        spans: list[tuple[int, int]] = []
        for concept in concepts:
            expanded = build_prompts(concept.phrase)
            spans.append((len(prompts), len(prompts) + len(expanded)))
            prompts.extend(expanded)

        with torch.no_grad():
            tokens = processor(text=prompts, return_tensors="pt", padding=True)
            tokens = {key: value.to(device) for key, value in tokens.items()}
            features = base.get_text_features(**tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            pooled = torch.stack(
                [features[start:end].mean(dim=0) for start, end in spans], dim=0
            )
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        return pooled

    return load_cached(cache_key, factory)


def score_concepts(
    image,
    concepts: tuple[Concept, ...] = CONCEPTS,
    variant: str = "baseline",
) -> ScoreResult:
    """Score an image against a concept vocabulary.

    Each concept is expanded into a template ensemble whose text embeddings are averaged
    before comparison - a standard zero-shot robustness technique. The returned
    probabilities are a softmax over the supplied concept set, so they are *relative to
    that set*; callers surface this in their confidence signals.

    Args:
        image: A PIL RGB image.
        concepts: Concepts to score. Defaults to the full vocabulary.
        variant: ``"baseline"`` or ``"adapted"`` (Phase 5 LoRA model).

    Raises:
        ModelUnavailableError: if the model or its dependencies are missing.
    """
    status = availability(variant)
    if not status.available:
        raise ModelUnavailableError(status.reason)

    import torch

    model, processor, device = _load(variant)
    base = _base_module(model)
    text_embeddings = _concept_text_embeddings(concepts, variant)

    with torch.no_grad():
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        image_features = base.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = base.logit_scale.exp() * image_features @ text_embeddings.t()
        probabilities = logits.softmax(dim=-1)[0].tolist()

    scored = [
        ConceptScore(key=concept.key, label=label_for(concept.key), probability=float(prob))
        for concept, prob in zip(concepts, probabilities, strict=True)
    ]
    scored.sort(key=lambda item: item.probability, reverse=True)
    return ScoreResult(
        scores=scored,
        model_id=CLIP_RS_MODEL_ID,
        device=device,
        variant=variant,
        version=model_version(variant),
    )


def score_land_cover(image, variant: str = "baseline") -> ScoreResult:
    """Score an image against land-cover concepts only (for 'dominant land cover')."""
    return score_concepts(image, LAND_COVER_CONCEPTS, variant=variant)


#: Minimum tile contrast required to report a coarse region. Below this the phrase does not
#: distinguish any part of the image, so no region is claimed.
MIN_PATCH_CONTRAST = 0.15


@dataclass
class PatchLocalization:
    """A coarse localization produced by scoring image tiles against one phrase."""

    #: ``(x_min, y_min, x_max, y_max)`` in the coordinates of the scored image.
    box: tuple[float, float, float, float]
    #: How strongly the winning tiles stand out from the typical tile, 0-1. Measured as the
    #: best tile's margin over the median tile, relative to the full tile spread. This is a
    #: real contrast measure, not a normalised maximum (which would always be 1.0).
    contrast: float
    #: Raw per-tile similarity for the winning tile.
    similarity: float
    #: Number of tiles merged into the reported region.
    tile_count: int
    grid: int
    model_id: str = CLIP_RS_MODEL_ID
    device: str = "cpu"
    variant: str = "baseline"


def localize_by_patches(
    image,
    phrase: str,
    grid: int = 4,
    top_tiles: int = 3,
    variant: str = "baseline",
) -> PatchLocalization | None:
    """Coarsely localize a phrase by scoring a grid of image tiles.

    Why this exists: open-vocabulary detectors are trained on natural photographs and
    frequently return nothing for diffuse overhead land cover such as water or farmland.
    Rather than reporting "not found" for a concept the scene clearly contains, the image is
    split into a grid, each tile is scored against the phrase with the remote-sensing CLIP
    model, and the highest-scoring tiles are merged into a region.

    This yields a genuine, measured region - but it is **coarse** (tile-resolution, not an
    object boundary), which callers state explicitly. Returns ``None`` when no tile stands
    out from the others.

    Args:
        image: A PIL RGB image.
        phrase: The concept phrase to localize.
        grid: Tiles per side.
        top_tiles: How many best tiles to merge into the reported region.

    Raises:
        ModelUnavailableError: if the model or its dependencies are missing.
    """
    status = availability(variant)
    if not status.available:
        raise ModelUnavailableError(status.reason)

    import torch

    model, processor, device = _load(variant)
    base = _base_module(model)

    width, height = image.size
    tile_w = width / grid
    tile_h = height / grid

    tiles = []
    boxes: list[tuple[float, float, float, float]] = []
    for row in range(grid):
        for col in range(grid):
            box = (col * tile_w, row * tile_h, (col + 1) * tile_w, (row + 1) * tile_h)
            boxes.append(box)
            tiles.append(image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3]))))

    prompts = build_prompts(phrase)

    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        text_features = base.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_vector = text_features.mean(dim=0, keepdim=True)
        text_vector = text_vector / text_vector.norm(dim=-1, keepdim=True)

        pixel_values = processor(images=tiles, return_tensors="pt")["pixel_values"].to(device)
        tile_features = base.get_image_features(pixel_values=pixel_values)
        tile_features = tile_features / tile_features.norm(dim=-1, keepdim=True)

        similarities = (tile_features @ text_vector.t()).squeeze(-1)

    values = similarities.tolist()
    best = max(values)
    worst = min(values)
    spread = best - worst
    # A flat response means the phrase does not distinguish any part of the image.
    if spread <= 1e-4:
        return None

    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )

    # Contrast: how far the best tile sits above a typical tile, relative to the spread.
    # Unlike a normalised maximum (always 1.0), this varies with how localized the
    # response actually is.
    contrast = (best - median) / spread
    if contrast < MIN_PATCH_CONTRAST:
        return None

    normalised = [(value - worst) / spread for value in values]
    ranked = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    chosen = [index for index in ranked[: max(1, top_tiles)] if normalised[index] >= 0.6]
    if not chosen:
        chosen = ranked[:1]

    x_min = min(boxes[index][0] for index in chosen)
    y_min = min(boxes[index][1] for index in chosen)
    x_max = max(boxes[index][2] for index in chosen)
    y_max = max(boxes[index][3] for index in chosen)

    return PatchLocalization(
        box=(x_min, y_min, x_max, y_max),
        contrast=float(min(1.0, max(0.0, contrast))),
        similarity=float(best),
        tile_count=len(chosen),
        grid=grid,
        model_id=CLIP_RS_MODEL_ID,
        device=device,
        variant=variant,
    )
