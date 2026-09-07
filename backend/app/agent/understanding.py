"""Query understanding and task classification.

Maps an untrusted natural-language query plus the current input configuration onto one
supported task. The approach is transparent keyword/pattern scoring constrained by the
actual input shape, because the input configuration is often a stronger signal than
wording (two co-located images almost certainly means a temporal or fusion task).

Design notes:
- Vocabulary is intentionally *general* (change/difference, water/built-up, highlight/
  locate, describe/caption) rather than tuned to demo phrasings, so unseen queries from a
  hidden evaluation set route sensibly.
- The query is treated as untrusted data. It is length-bounded and control characters are
  stripped. Instruction-like text is neutralized (see ``sanitize_query``); it never
  influences which tools exist or what parameters are allowed - that comes only from the
  registry.
"""

from __future__ import annotations

import re
import unicodedata

from ..registry.tasks import TASK_REGISTRY, Modality, TaskType
from .types import InputConfiguration, QueryIntent

MAX_QUERY_LENGTH = 1000

# Patterns that look like attempts to redirect the system rather than ask about imagery.
# Matching text is recorded as a blocked signal and stripped before classification; it
# can never change tool availability or parameters (those come from the registry only).
_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(
        r"disregard\s+(all\s+|any\s+)?(previous|prior|above)\s+(instructions?|rules?)", re.I
    ),
    re.compile(r"you\s+are\s+now\s+a\b", re.I),
    re.compile(r"\bsystem\s*prompt\b", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions?|reasoning)", re.I),
    re.compile(r"\b(chain[-\s]?of[-\s]?thought)\b", re.I),
    re.compile(r"\bexecute\b.*\b(command|shell|script)\b", re.I),
    re.compile(r"\b(rm\s+-rf|drop\s+table|;\s*shutdown)\b", re.I),
)

# --- Task vocabulary ------------------------------------------------------------
# Weights are relative; absolute values only matter for ranking.
_CHANGE_TERMS = (
    "change",
    "changed",
    "changes",
    "difference",
    "differences",
    "differ",
    "between these",
    "between the two",
    "over time",
    "since",
    "before and after",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "grown",
    "growth",
    "expanded",
    "expansion",
    "shrunk",
    "deforestation",
    "urbanization",
    "new construction",
    "compare",
    "comparison",
)

_GROUNDING_TERMS = (
    "highlight",
    "locate",
    "where is",
    "where are",
    "show me the",
    "mark",
    "outline",
    "delineate",
    "segment",
    "find the",
    "identify the location",
    "point out",
    "bounding box",
)

_CAPTION_TERMS = (
    "describe",
    "description",
    "caption",
    "summarize",
    "summarise",
    "overview",
    "what does this image show",
    "what is in this image",
    "tell me about this image",
    "characterize",
)

_FUSION_TERMS = (
    "fuse",
    "fusion",
    "combine",
    "combined",
    "together",
    "jointly",
    "joint",
    "both images",
    "both modalities",
    "optical and sar",
    "sar and optical",
    "complementary",
    # Cross-modal questions that name the modalities or radar. These route to fusion so a
    # question about what one modality adds over the other is not misread as single-image VQA.
    "sar reveal",
    "sar provide",
    "sar show",
    "radar reveal",
    "than optical",
    "in optical",
    "which modality",
    "between optical",
    "backscatter",
    "cross-modal",
    "cross modal",
)

_METADATA_TERMS = (
    "metadata",
    "crs",
    "coordinate reference",
    "projection",
    "epsg",
    "resolution",
    "bounds",
    "bounding coordinates",
    "extent",
    "how many bands",
    "band count",
    "pixel size",
    "georeference",
    "georeferenced",
    "data type",
    "dtype",
    "file info",
    "image size",
    "dimensions",
)

# Question shapes that indicate VQA when no more specific task matches.
_QUESTION_TERMS = (
    "what",
    "is there",
    "are there",
    "does",
    "do ",
    "how much",
    "how many",
    "which",
    "any ",
    "present",
    "visible",
    "dominant",
    "dominates",
)

_TARGET_PATTERNS = (
    re.compile(
        r"(?:highlight|locate|outline|delineate|segment|mark|show me)"
        r"\s+(?:the\s+|all\s+)?([a-z0-9 \-]{3,40})",
        re.I,
    ),
    re.compile(r"where\s+(?:is|are)\s+(?:the\s+)?([a-z0-9 \-]{3,40})", re.I),
    re.compile(r"find\s+(?:the\s+)?([a-z0-9 \-]{3,40})", re.I),
)


def sanitize_query(raw: str) -> tuple[str, list[str]]:
    """Normalize an untrusted query.

    Returns ``(clean_query, blocked_signals)``. Control characters are removed, unicode
    is normalized, length is bounded, and instruction-like fragments are stripped and
    reported so the caller can log/surface them.
    """
    text = unicodedata.normalize("NFKC", raw or "")
    text = "".join(
        ch
        for ch in text
        if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
    )
    text = text.replace("\n", " ").replace("\t", " ")

    blocked: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            blocked.append(pattern.pattern)
            text = pattern.sub(" ", text)

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_QUERY_LENGTH:
        text = text[:MAX_QUERY_LENGTH].rstrip()
    return text, blocked


def _count_hits(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def extract_target(query: str) -> str | None:
    """Extract the object a grounding query refers to, if any."""
    for pattern in _TARGET_PATTERNS:
        match = pattern.search(query)
        if match:
            target = match.group(1).strip().rstrip("?.,")
            # Drop trailing filler that is not part of the target noun phrase.
            target = re.sub(
                r"\b(in|on|of|from|within)\s+(this|the)\s+(image|scene|raster).*$",
                "",
                target,
            ).strip()
            if len(target) >= 3:
                return target
    return None


def _task_is_feasible(task: TaskType, inputs: InputConfiguration) -> bool:
    """Whether the input configuration can satisfy a task's declared shape."""
    spec = TASK_REGISTRY.get(task)
    if spec is None:
        return False
    if not (spec.min_images <= inputs.count <= spec.max_images):
        return False
    if task == TaskType.OPTICAL_SAR_FUSION and not inputs.has_optical_sar_pair():
        return False
    if task in (TaskType.TEMPORAL_CHANGE, TaskType.CHANGE_VQA) and inputs.count != 2:
        return False
    return True


def classify(query: str, inputs: InputConfiguration) -> QueryIntent:
    """Classify a sanitized query into a task, constrained by the input configuration.

    The input shape acts as a hard filter: a task is only selectable when the supplied
    images can actually satisfy it. Wording then ranks the feasible candidates.
    """
    text = query.lower().strip()
    scores: dict[TaskType, float] = {}
    signals: dict[TaskType, list[str]] = {}

    def add(task: TaskType, weight: float, hits: list[str]) -> None:
        if not hits and weight <= 0:
            return
        scores[task] = scores.get(task, 0.0) + weight
        signals.setdefault(task, []).extend(hits)

    two_images = inputs.count == 2
    is_question = text.endswith("?") or bool(_count_hits(text, _QUESTION_TERMS))

    # --- wording signals --------------------------------------------------------
    metadata_hits = _count_hits(text, _METADATA_TERMS)
    if metadata_hits:
        add(TaskType.METADATA_INSPECTION, 2.0 + 0.3 * len(metadata_hits), metadata_hits)

    change_hits = _count_hits(text, _CHANGE_TERMS)
    if change_hits:
        # A change question resolves to CHANGE_VQA; an open request to CHANGE description.
        weight = 2.0 + 0.3 * len(change_hits)
        if is_question:
            add(TaskType.CHANGE_VQA, weight, change_hits)
            add(TaskType.TEMPORAL_CHANGE, weight - 0.5, change_hits)
        else:
            add(TaskType.TEMPORAL_CHANGE, weight, change_hits)
            add(TaskType.CHANGE_VQA, weight - 0.5, change_hits)

    fusion_hits = _count_hits(text, _FUSION_TERMS)
    if fusion_hits:
        add(TaskType.OPTICAL_SAR_FUSION, 2.0 + 0.4 * len(fusion_hits), fusion_hits)

    grounding_hits = _count_hits(text, _GROUNDING_TERMS)
    if grounding_hits:
        add(TaskType.GROUNDING, 2.2 + 0.3 * len(grounding_hits), grounding_hits)

    caption_hits = _count_hits(text, _CAPTION_TERMS)
    if caption_hits:
        add(TaskType.CAPTIONING, 2.0 + 0.3 * len(caption_hits), caption_hits)

    # A generic question falls back to VQA - but only when no more specific intent was
    # detected. In particular, a change or fusion request must NOT degrade into VQA,
    # because answering a different question than the one asked is worse than refusing.
    if is_question and not (
        grounding_hits or caption_hits or metadata_hits or change_hits or fusion_hits
    ):
        question_hits = _count_hits(text, _QUESTION_TERMS) or ["question form"]
        add(TaskType.VQA, 1.5, question_hits)
        if two_images:
            add(TaskType.CHANGE_VQA, 0.6, question_hits)

    # --- input-shape signals ----------------------------------------------------
    # Two images strongly imply a pair task even when the wording is vague.
    if two_images:
        if inputs.has_optical_sar_pair():
            add(TaskType.OPTICAL_SAR_FUSION, 1.2, ["optical+SAR pair supplied"])
        if inputs.has_temporal_pair():
            add(TaskType.TEMPORAL_CHANGE, 1.0, ["T1/T2 pair supplied"])
            add(TaskType.CHANGE_VQA, 0.8, ["T1/T2 pair supplied"])

    # --- feasibility filter -----------------------------------------------------
    # Infeasible candidates are filtered rather than deleted above, so that when nothing
    # is feasible we can still say *what* the user appeared to ask for.
    feasible = {
        task: score for task, score in scores.items() if _task_is_feasible(task, inputs)
    }

    if not feasible:
        if inputs.count == 0:
            return QueryIntent(
                task=TaskType.UNSUPPORTED,
                score=0.0,
                reason="No imagery was supplied, so there is nothing to analyze.",
            )
        # A recognizable request that the supplied images cannot satisfy.
        if scores:
            requested = max(scores, key=lambda t: scores[t])
            spec = TASK_REGISTRY[requested]
            return QueryIntent(
                task=TaskType.UNSUPPORTED,
                score=0.0,
                reason=(
                    f"The query looks like a {spec.title} request, but that needs "
                    f"{spec.min_images}-{spec.max_images} image(s) of the right modality "
                    f"and {inputs.count} was supplied."
                ),
            )
        return QueryIntent(
            task=TaskType.UNSUPPORTED,
            score=0.0,
            reason=(
                "The query could not be matched to a supported analysis task. Try asking "
                "about land cover, water, built-up areas, changes between two dates, or "
                "image metadata."
            ),
        )

    ranked = sorted(feasible.items(), key=lambda item: item[1], reverse=True)
    best_task, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    # Normalize into 0-1: absolute strength, discounted when a rival scores similarly.
    strength = min(1.0, best_score / 3.5)
    margin = 1.0 if best_score <= 0 else min(1.0, (best_score - runner_up) / best_score + 0.4)
    score = round(max(0.05, min(1.0, strength * margin)), 3)

    return QueryIntent(
        task=best_task,
        score=score,
        matched_signals=sorted(set(signals.get(best_task, [])))[:8],
        target=extract_target(query) if best_task == TaskType.GROUNDING else None,
        reason=f"Matched {TASK_REGISTRY[best_task].title} for {inputs.count} input image(s).",
        alternatives=[task for task, _ in ranked[1:3]],
    )


def understand(raw_query: str, inputs: InputConfiguration) -> tuple[QueryIntent, str, list[str]]:
    """Sanitize then classify a query.

    Returns ``(intent, sanitized_query, blocked_signals)``.
    """
    clean, blocked = sanitize_query(raw_query)
    if not clean:
        return (
            QueryIntent(
                task=TaskType.UNSUPPORTED,
                score=0.0,
                reason="The query was empty after validation.",
            ),
            clean,
            blocked,
        )
    return classify(clean, inputs), clean, blocked


def required_modalities(task: TaskType) -> list[Modality]:
    """Modalities a task declares support for (from the task registry)."""
    spec = TASK_REGISTRY.get(task)
    return list(spec.modalities) if spec else []
