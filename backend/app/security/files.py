"""Upload safety: filename sanitization, path confinement, and content-type checks.

Every uploaded file is treated as untrusted. This module never trusts a
client-supplied filename or path; it derives a safe storage name and confirms that
every resolved path stays inside the configured storage root.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from pathlib import Path

from geospatial import SUPPORTED_EXTENSIONS

# Content types browsers/clients commonly send for supported rasters. The declared
# type is advisory only - the authoritative check is opening the file as a raster.
ALLOWED_CONTENT_TYPES = {
    "image/tiff",
    "image/tif",
    "image/geotiff",
    "image/x-tiff",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "application/octet-stream",  # frequently used for .tif uploads
    "",  # some clients omit the header entirely
}

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_STEM_LENGTH = 64


class UnsafePathError(ValueError):
    """Raised when a resolved path would escape its permitted root."""


def sanitize_filename(filename: str) -> str:
    """Reduce an untrusted filename to a safe ``stem.ext`` form.

    Strips directory components (defeating ``../`` and absolute-path tricks), removes
    control/unicode trickery, collapses unsafe characters, and bounds the length.
    Returns ``"upload"`` (no extension) when nothing usable remains.
    """
    # Normalize unicode so lookalike/combining characters cannot smuggle separators.
    normalized = unicodedata.normalize("NFKD", filename or "")
    normalized = normalized.replace("\x00", "")

    # Discard any directory part, handling both POSIX and Windows separators.
    base = normalized.replace("\\", "/").split("/")[-1].strip()

    suffix = Path(base).suffix.lower()
    stem = Path(base).stem

    stem = _UNSAFE_CHARS.sub("_", stem).strip("._-")
    if not stem:
        stem = "upload"
    stem = stem[:_MAX_STEM_LENGTH]

    if suffix not in SUPPORTED_EXTENSIONS:
        # Do not preserve unknown/dangerous extensions; validation rejects these later.
        return stem

    return f"{stem}{suffix}"


def build_storage_name(original_filename: str, asset_id: str | None = None) -> str:
    """Build a collision-free storage filename that keeps a readable hint of the original."""
    safe = sanitize_filename(original_filename)
    unique = asset_id or uuid.uuid4().hex[:12]
    suffix = Path(safe).suffix
    stem = Path(safe).stem or "upload"
    return f"{unique}_{stem}{suffix}"


def resolve_within(root: Path, *parts: str) -> Path:
    """Join ``parts`` under ``root`` and confirm the result stays inside ``root``.

    Raises:
        UnsafePathError: if the resolved path escapes ``root``.
    """
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*parts).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise UnsafePathError("Resolved path escapes the permitted storage root.")
    return candidate


def is_allowed_content_type(content_type: str | None) -> bool:
    """Check a declared content type against the allowlist (advisory check)."""
    if content_type is None:
        return True
    return content_type.split(";")[0].strip().lower() in ALLOWED_CONTENT_TYPES
