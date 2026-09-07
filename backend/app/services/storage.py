"""Filesystem storage for uploads and derived artifacts.

Layout under the configured storage root::

    uploads/    original uploaded rasters
    processed/  derived rasters
    evidence/   previews and analysis evidence images
    reports/    generated reports
    temp/       scratch space

All writes go through :func:`save_upload_stream`, which enforces the size limit while
streaming so an oversized file is never fully buffered in memory or left on disk.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ..core.config import get_settings
from ..security.files import resolve_within

UPLOADS_DIR = "uploads"
PROCESSED_DIR = "processed"
EVIDENCE_DIR = "evidence"
REPORTS_DIR = "reports"
TEMP_DIR = "temp"

_ALL_DIRS = (UPLOADS_DIR, PROCESSED_DIR, EVIDENCE_DIR, REPORTS_DIR, TEMP_DIR)

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class UploadTooLargeError(ValueError):
    """Raised when an upload exceeds the configured size limit."""


@dataclass(frozen=True)
class StoredFile:
    """A file written into the storage root."""

    absolute_path: Path
    relative_path: str
    size_bytes: int


def storage_root() -> Path:
    """Return the storage root, creating the standard subdirectories if needed."""
    root = get_settings().storage_path
    for name in _ALL_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def absolute_path(relative_path: str) -> Path:
    """Resolve a stored relative path back to an absolute path inside the root.

    Raises:
        UnsafePathError: if the path escapes the storage root.
    """
    return resolve_within(storage_root(), *Path(relative_path).parts)


def save_upload_stream(
    source: BinaryIO,
    filename: str,
    subdir: str = UPLOADS_DIR,
    max_bytes: int | None = None,
) -> StoredFile:
    """Stream ``source`` to ``subdir/filename`` enforcing a maximum size.

    ``filename`` must already be sanitized (see ``security.files``). Raises
    :class:`UploadTooLargeError` and removes the partial file when the limit is hit.
    """
    settings = get_settings()
    limit = max_bytes if max_bytes is not None else settings.max_upload_mb * 1024 * 1024

    root = storage_root()
    target = resolve_within(root, subdir, filename)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    try:
        with target.open("wb") as sink:
            while True:
                chunk = source.read(_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise UploadTooLargeError(
                        f"The file exceeds the maximum upload size of {settings.max_upload_mb} MB."
                    )
                sink.write(chunk)
    except BaseException:
        target.unlink(missing_ok=True)
        raise

    return StoredFile(
        absolute_path=target,
        relative_path=target.relative_to(root).as_posix(),
        size_bytes=written,
    )


def delete_stored(relative_path: str) -> bool:
    """Delete a stored file. Returns True when a file was removed."""
    try:
        path = absolute_path(relative_path)
    except ValueError:
        return False
    if path.is_file():
        path.unlink()
        return True
    return False


def clear_temp() -> None:
    """Remove everything under the temp directory."""
    temp = storage_root() / TEMP_DIR
    if temp.is_dir():
        shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir(parents=True, exist_ok=True)
