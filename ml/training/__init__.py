"""Training and adaptation pipelines.

Separable from the serving backend so GPU work can run wherever a GPU is available without
changing the API.
"""

from .lora_clip import (
    CLIP_MAX_TEXT_TOKENS,
    DEFAULT_TARGET_MODULES,
    CaptionDataset,
    LoraTrainingConfig,
    TrainingMetrics,
    attach_lora,
    build_collate_fn,
    contrastive_loss,
    evaluate_loss,
    load_adapted_clip,
    load_split,
    set_seed,
    train_lora_clip,
)

__all__ = [
    "CLIP_MAX_TEXT_TOKENS",
    "CaptionDataset",
    "DEFAULT_TARGET_MODULES",
    "LoraTrainingConfig",
    "TrainingMetrics",
    "attach_lora",
    "build_collate_fn",
    "contrastive_loss",
    "evaluate_loss",
    "load_adapted_clip",
    "load_split",
    "set_seed",
    "train_lora_clip",
]
