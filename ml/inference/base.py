"""Model loading, caching, and availability reporting.

Design points:

- **Lazy**: nothing is imported or downloaded until a backend is actually used, so the
  API starts fast and the geospatial layer works without the ML extras installed.
- **Cached**: a loaded model is reused for the process lifetime. Reloading a 150M-parameter
  model per request would dominate latency.
- **Honest about availability**: :func:`describe_availability` reports whether torch and
  the weights are actually present. Nothing pretends to be loadable when it is not.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

# Model identifiers are declared here so the registry, docs, and code cannot drift.
CLIP_RS_MODEL_ID = "flax-community/clip-rsicd-v2"
OWLVIT_MODEL_ID = "google/owlvit-base-patch32"

_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()


class ModelUnavailableError(RuntimeError):
    """Raised when a model cannot be loaded (missing dependency or missing weights)."""


@dataclass(frozen=True)
class Availability:
    """Whether a backend can run right now, and why not if it cannot."""

    available: bool
    reason: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.available


def torch_available() -> bool:
    """True when PyTorch can be imported."""
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


def transformers_available() -> bool:
    """True when Transformers can be imported."""
    try:
        import transformers  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def resolve_device(preference: str | None = None) -> str:
    """Resolve the compute device.

    ``auto`` selects CUDA only when it is genuinely usable; otherwise CPU. An explicit
    ``cuda`` request falls back to CPU with no pretence when CUDA is absent.
    """
    requested = (preference or os.environ.get("DEVICE") or "cpu").strip().lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def is_cached(model_id: str) -> bool:
    """True when weights for ``model_id`` are present in the local Hugging Face cache.

    Used to decide whether a model can load without network access, which is what makes
    offline test skipping and honest status reporting possible.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached, or hub unavailable
        return False
    return True


def describe_availability(model_id: str, *, allow_download: bool | None = None) -> Availability:
    """Report whether a model backend can run, with a human-readable reason if not."""
    if not torch_available():
        return Availability(False, "PyTorch is not installed.")
    if not transformers_available():
        return Availability(False, "Transformers is not installed.")

    if allow_download is None:
        allow_download = os.environ.get("HF_HUB_OFFLINE", "").strip() not in {"1", "true", "True"}

    if not is_cached(model_id) and not allow_download:
        return Availability(
            False,
            f"Weights for '{model_id}' are not cached and downloads are disabled.",
        )
    return Availability(True)


def load_cached(key: str, factory) -> Any:
    """Return a cached object, building it once under a lock."""
    if key in _CACHE:
        return _CACHE[key]
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = factory()
    return _CACHE[key]


def clear_cache() -> None:
    """Drop cached models. Used by tests and to release memory."""
    with _LOCK:
        _CACHE.clear()
