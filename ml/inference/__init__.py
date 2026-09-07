"""Inference backends for the single-image specialists.

Two real, CPU-runnable models back Phase 4:

- ``flax-community/clip-rsicd-v2`` - CLIP fine-tuned on RSICD (remote-sensing captions),
  used for zero-shot concept scoring behind VQA and captioning.
- ``google/owlvit-base-patch32`` - open-vocabulary detection, used for grounding.

Both are baselines: real inference, real scores, but uncalibrated and not yet adapted with
LoRA on remote-sensing data (that is Phase 5). Imports are lazy so the geospatial layer and
the API remain usable without the ML extras installed.

Phase 6 adds :mod:`ml.inference.temporal`, which is **algorithmic rather than learned**:
change detection by radiometrically normalised change vector analysis. It needs neither
torch nor model weights, so temporal change works on any machine. Its symbols are exported
here too, and importing this package does not pull torch.
"""

from .adapted_clip import (
    DEFAULT_ADAPTER_DIR,
    AdapterInfo,
    ClipVariant,
    adapter_dir,
    describe_adapter,
    resolve_variant,
    variant_version,
)
from .base import (
    CLIP_RS_MODEL_ID,
    OWLVIT_MODEL_ID,
    Availability,
    ModelUnavailableError,
    clear_cache,
    describe_availability,
    is_cached,
    resolve_device,
    torch_available,
    transformers_available,
)
from .fusion import (
    FUSION_ALGORITHM_VERSION,
    ComplementaryRegion,
    CrossModalQuestion,
    CrossModalResult,
    ModalityFeatures,
    WaterAgreement,
    analyze_cross_modal,
    classify_cross_modal_question,
    cross_modal_stats,
)
from .temporal import (
    ALGORITHM_VERSION,
    ChangeConfig,
    ChangeMethod,
    ChangeRegion,
    ChangeResult,
    IndexChange,
    Registration,
    detect_change,
    otsu_threshold,
    radiometric_normalize,
    robust_spread,
)
from .vocabulary import (
    CONCEPTS,
    CONCEPTS_BY_KEY,
    LAND_COVER_CONCEPTS,
    PROMPT_TEMPLATES,
    Concept,
    QuestionAnalysis,
    QuestionKind,
    analyze_question,
    detection_phrase,
    find_concepts,
    label_for,
)

__all__ = [
    "ALGORITHM_VERSION",
    "FUSION_ALGORITHM_VERSION",
    "AdapterInfo",
    "Availability",
    "CLIP_RS_MODEL_ID",
    "CONCEPTS",
    "CONCEPTS_BY_KEY",
    "ChangeConfig",
    "ChangeMethod",
    "ChangeRegion",
    "ChangeResult",
    "ClipVariant",
    "ComplementaryRegion",
    "Concept",
    "CrossModalQuestion",
    "CrossModalResult",
    "DEFAULT_ADAPTER_DIR",
    "IndexChange",
    "LAND_COVER_CONCEPTS",
    "ModalityFeatures",
    "ModelUnavailableError",
    "OWLVIT_MODEL_ID",
    "Registration",
    "WaterAgreement",
    "adapter_dir",
    "analyze_cross_modal",
    "classify_cross_modal_question",
    "cross_modal_stats",
    "describe_adapter",
    "detect_change",
    "otsu_threshold",
    "radiometric_normalize",
    "resolve_variant",
    "robust_spread",
    "variant_version",
    "PROMPT_TEMPLATES",
    "QuestionAnalysis",
    "QuestionKind",
    "analyze_question",
    "clear_cache",
    "describe_availability",
    "detection_phrase",
    "find_concepts",
    "is_cached",
    "label_for",
    "resolve_device",
    "torch_available",
    "transformers_available",
]
