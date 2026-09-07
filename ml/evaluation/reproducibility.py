"""Reproducibility metadata for benchmark runs.

Every benchmark writes a ``config.json`` describing exactly what produced its numbers:
dataset identity, seed, model/tool versions, library versions, device, git commit, and a
timestamp. If the repo has no commit yet the commit is reported as ``HEAD / uncommitted`` —
never a fabricated hash.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _safe_version(module_name: str) -> str | None:
    """Return an installed package version, or None if it is not importable."""
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:  # noqa: BLE001 - a missing optional dependency is not fatal here
        return None


def git_commit(repo_root: Path) -> str:
    """Return the current commit hash, or ``HEAD / uncommitted`` when there is none.

    Never fabricates: if git is unavailable or the repo has no commits, the honest
    placeholder is returned so a reader knows the state is not pinned.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        out = result.stdout.strip()
        if result.returncode == 0 and out:
            return out
    except Exception:  # noqa: BLE001 - git may be absent; fall through to the placeholder
        pass
    return "HEAD / uncommitted"


def device_summary() -> dict[str, Any]:
    """Describe the compute device without requiring torch."""
    info: dict[str, Any] = {"device": "cpu"}
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_bytes"] = int(props.total_memory)
    except Exception:  # noqa: BLE001 - torch optional for algorithmic benchmarks
        info["torch_cuda_available"] = False
    return info


@dataclass
class RunContext:
    """Environment + configuration snapshot for one benchmark run."""

    benchmark: str
    dataset: str
    split: str
    samples: int
    seed: int
    timestamp_utc: str = ""
    python: str = ""
    platform: str = ""
    numpy: str | None = None
    torch: str | None = None
    transformers: str | None = None
    peft: str | None = None
    device: dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    models: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_context(
    *,
    benchmark: str,
    dataset: str,
    split: str,
    samples: int,
    seed: int,
    repo_root: Path,
    models: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> RunContext:
    """Assemble a fully-populated :class:`RunContext`."""
    return RunContext(
        benchmark=benchmark,
        dataset=dataset,
        split=split,
        samples=samples,
        seed=seed,
        timestamp_utc=datetime.now(UTC).isoformat(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        numpy=_safe_version("numpy"),
        torch=_safe_version("torch"),
        transformers=_safe_version("transformers"),
        peft=_safe_version("peft"),
        device=device_summary(),
        git_commit=git_commit(repo_root),
        models=models or {},
        extra=extra or {},
    )


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, for dataset/manifest integrity records."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    """Write ``payload`` as pretty JSON, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    """Write an iterable of records as JSON Lines."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
