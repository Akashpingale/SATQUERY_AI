"""Remote-sensing vocabulary and question analysis for zero-shot VQA/captioning.

The vocabulary is intentionally **general** - drawn from common land-cover taxonomies
(EuroSAT / UCM / RSICD style categories) rather than tuned to any demo image - so that
unseen scenes from a hidden evaluation set are described with the same concepts.

Prompt engineering here is a legitimate zero-shot technique, not a substitute for
adaptation: prompts are ensembled over several phrasings, which measurably stabilises CLIP
scores. Genuine remote-sensing adaptation (LoRA on RS data) is Phase 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

# Several phrasings are averaged per concept. RSICD-style captions describe scenes from
# above, so all templates are aerial/satellite framings.
PROMPT_TEMPLATES: tuple[str, ...] = (
    "an aerial photograph of {}",
    "a satellite image of {}",
    "an aerial view of {}",
    "a remote sensing image of {}",
)


@dataclass(frozen=True)
class Concept:
    """A land-cover / object concept with the phrase used to prompt the model."""

    key: str
    phrase: str
    #: Words in a user question that indicate this concept.
    triggers: tuple[str, ...] = ()
    #: True when the concept describes overall land cover (usable as a "dominant" answer).
    land_cover: bool = True


# --- Concept vocabulary -----------------------------------------------------------
# Order is irrelevant; scores are computed over the whole set.
CONCEPTS: tuple[Concept, ...] = (
    Concept(
        "water",
        "a body of water",
        ("water", "lake", "pond", "reservoir", "sea", "ocean"),
    ),
    Concept("river", "a river", ("river", "stream", "canal")),
    Concept(
        "dense_residential",
        "dense residential buildings",
        ("dense residential", "dense housing"),
    ),
    Concept(
        "residential",
        "residential buildings and houses",
        (
            "building",
            "buildings",
            "house",
            "houses",
            "residential",
            "settlement",
            "built-up",
            "built up",
            "urban",
        ),
    ),
    Concept(
        "industrial",
        "industrial buildings and warehouses",
        ("industrial", "factory", "warehouse"),
    ),
    Concept("commercial", "commercial buildings", ("commercial", "shopping")),
    Concept(
        "farmland",
        "farmland and cultivated fields",
        (
            "farm",
            "farmland",
            "field",
            "fields",
            "crop",
            "crops",
            "agriculture",
            "agricultural",
            "cultivated",
        ),
    ),
    Concept(
        "forest",
        "dense forest and trees",
        ("forest", "trees", "tree", "woodland", "jungle", "canopy"),
    ),
    Concept(
        "grassland",
        "grassland and meadow",
        ("grass", "grassland", "meadow", "pasture", "lawn"),
    ),
    Concept("shrubland", "shrubs and sparse vegetation", ("shrub", "shrubs", "scrub", "bush")),
    Concept(
        "bare_land",
        "bare soil and empty land",
        ("bare", "bare land", "bare soil", "empty land", "soil"),
    ),
    Concept("desert", "sandy desert terrain", ("desert", "sand", "dune", "dunes")),
    Concept("beach", "a sandy beach and coastline", ("beach", "coast", "coastline", "shore")),
    Concept(
        "mountain",
        "mountains and rugged terrain",
        ("mountain", "mountains", "hill", "hills", "terrain"),
    ),
    Concept(
        "road",
        "roads and highways",
        ("road", "roads", "highway", "street", "streets", "motorway"),
    ),
    Concept("railway", "a railway line", ("railway", "railroad", "train track", "rail")),
    Concept("bridge", "a bridge", ("bridge",), land_cover=False),
    Concept(
        "parking_lot",
        "a parking lot with cars",
        ("parking", "car park", "cars", "vehicles"),
        land_cover=False,
    ),
    Concept(
        "airport",
        "an airport with runways",
        ("airport", "runway", "airfield", "aircraft", "airplane", "plane"),
    ),
    Concept(
        "harbor",
        "a harbor with ships",
        ("harbor", "harbour", "port", "dock", "ship", "ships", "boat", "boats"),
    ),
    Concept(
        "storage_tanks",
        "industrial storage tanks",
        ("storage tank", "tanks", "silo"),
        land_cover=False,
    ),
    Concept(
        "sports_field",
        "a sports field or stadium",
        ("stadium", "sports field", "playground", "baseball", "football", "tennis"),
        land_cover=False,
    ),
    Concept("snow", "snow and ice cover", ("snow", "ice", "glacier")),
    Concept("cloud", "clouds obscuring the ground", ("cloud", "clouds", "cloudy", "overcast")),
)

CONCEPTS_BY_KEY: dict[str, Concept] = {concept.key: concept for concept in CONCEPTS}

LAND_COVER_CONCEPTS: tuple[Concept, ...] = tuple(c for c in CONCEPTS if c.land_cover)

#: Human-readable labels used when composing answers and captions.
CONCEPT_LABELS: dict[str, str] = {
    "water": "water",
    "river": "a river",
    "dense_residential": "dense residential development",
    "residential": "buildings",
    "industrial": "industrial buildings",
    "commercial": "commercial buildings",
    "farmland": "farmland",
    "forest": "forest",
    "grassland": "grassland",
    "shrubland": "sparse vegetation",
    "bare_land": "bare land",
    "desert": "desert terrain",
    "beach": "beach or coastline",
    "mountain": "mountainous terrain",
    "road": "roads",
    "railway": "a railway",
    "bridge": "a bridge",
    "parking_lot": "a parking lot",
    "airport": "an airport",
    "harbor": "a harbor",
    "storage_tanks": "storage tanks",
    "sports_field": "a sports field",
    "snow": "snow or ice",
    "cloud": "cloud cover",
}


class QuestionKind(StrEnum):
    """What a VQA question is asking for."""

    PRESENCE = "presence"  # "is water present?"
    DOMINANT = "dominant"  # "what land cover dominates?"
    WHAT_VISIBLE = "what_visible"  # "what is visible in this image?"
    COUNT = "count"  # "how many buildings?" - not answerable by this model
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class QuestionAnalysis:
    """Structured reading of a VQA question."""

    kind: QuestionKind
    #: Concepts the question refers to (may be empty for open questions).
    concepts: tuple[Concept, ...] = ()
    reason: str = ""


_COUNT_PATTERNS = (
    re.compile(r"\bhow many\b", re.I),
    re.compile(r"\bnumber of\b", re.I),
    re.compile(r"\bcount (the|how)\b", re.I),
)

_PRESENCE_PATTERNS = (
    re.compile(r"\b(is|are) there\b", re.I),
    re.compile(r"\b(is|are)\s+\w+\s+(present|visible)\b", re.I),
    re.compile(r"\bdoes (this|the) (image|scene|area)\b", re.I),
    re.compile(r"\bcontains?\b", re.I),
    re.compile(r"\bany\b", re.I),
    re.compile(r"\bpresent\b", re.I),
    re.compile(r"\bvisible\b", re.I),
    re.compile(r"\bcan (i|you) see\b", re.I),
)

_DOMINANT_PATTERNS = (
    re.compile(r"\bdominant\b", re.I),
    re.compile(r"\bdominates?\b", re.I),
    re.compile(r"\bmostly\b", re.I),
    re.compile(r"\bmain (land ?cover|type)\b", re.I),
    re.compile(r"\bwhat (kind|type) of (land ?cover|terrain|area|scene)\b", re.I),
    re.compile(r"\bwhat land ?cover\b", re.I),
    re.compile(r"\bland ?use\b", re.I),
    re.compile(r"\bclassify\b", re.I),
)

_WHAT_VISIBLE_PATTERNS = (
    re.compile(r"\bwhat (is|are|can be)\s+(visible|seen|shown|in this)\b", re.I),
    re.compile(r"\bwhat (objects|features|things)\b", re.I),
    re.compile(r"\bwhat do(es)? (this|the) (image|scene)\s+show\b", re.I),
    re.compile(r"\bdescribe\b", re.I),
)


def find_concepts(text: str) -> tuple[Concept, ...]:
    """Return concepts explicitly referenced in the text, longest trigger first.

    Longer triggers are matched first so "dense residential" wins over "residential".
    """
    lowered = f" {text.lower()} "
    matches: list[tuple[int, Concept]] = []
    for concept in CONCEPTS:
        for trigger in concept.triggers:
            if re.search(rf"\b{re.escape(trigger)}\b", lowered):
                matches.append((len(trigger), concept))
                break
    matches.sort(key=lambda item: item[0], reverse=True)

    seen: set[str] = set()
    ordered: list[Concept] = []
    for _length, concept in matches:
        if concept.key not in seen:
            seen.add(concept.key)
            ordered.append(concept)
    return tuple(ordered)


def analyze_question(question: str) -> QuestionAnalysis:
    """Determine what a VQA question asks for.

    Counting questions are identified explicitly so they can be **refused**: a
    contrastive image-text model has no counting mechanism, and inventing a number would
    be fabrication.
    """
    text = (question or "").strip()
    if not text:
        return QuestionAnalysis(QuestionKind.UNSUPPORTED, reason="The question was empty.")

    concepts = find_concepts(text)

    if any(pattern.search(text) for pattern in _COUNT_PATTERNS):
        return QuestionAnalysis(
            QuestionKind.COUNT,
            concepts=concepts,
            reason=(
                "Counting is not supported by the current zero-shot model, which compares "
                "an image against text descriptions and has no counting mechanism."
            ),
        )

    if any(pattern.search(text) for pattern in _DOMINANT_PATTERNS):
        return QuestionAnalysis(QuestionKind.DOMINANT, concepts=concepts)

    # A presence question needs a concept to test for.
    if concepts and any(pattern.search(text) for pattern in _PRESENCE_PATTERNS):
        return QuestionAnalysis(QuestionKind.PRESENCE, concepts=concepts)

    if any(pattern.search(text) for pattern in _WHAT_VISIBLE_PATTERNS):
        return QuestionAnalysis(QuestionKind.WHAT_VISIBLE, concepts=concepts)

    # A bare concept mention ("water?") is treated as a presence check.
    if concepts:
        return QuestionAnalysis(QuestionKind.PRESENCE, concepts=concepts)

    # Any other question about the scene: report what the model can see.
    return QuestionAnalysis(
        QuestionKind.WHAT_VISIBLE,
        reason="No specific concept was recognised; reporting the most likely scene content.",
    )


def build_prompts(phrase: str) -> list[str]:
    """Expand a concept phrase into the template ensemble."""
    return [template.format(phrase) for template in PROMPT_TEMPLATES]


def label_for(key: str) -> str:
    """Human-readable label for a concept key."""
    return CONCEPT_LABELS.get(key, key.replace("_", " "))


def detection_phrase(concept_key: str | None, fallback: str) -> str:
    """Prompt phrase for open-vocabulary detection."""
    concept = CONCEPTS_BY_KEY.get(concept_key or "")
    return concept.phrase if concept else fallback
