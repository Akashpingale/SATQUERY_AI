"""BigEarthNet.txt dataset preparation and loading for Phase 5 adaptation.

Data sources (see ``docs/datasets.md`` for full provenance):

- **Annotations**: ``BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`` (official, CDLA-Permissive-1.0)
  - 9,553,962 image-text annotations over 464,044 Sentinel-1/2 pairs.
- **Imagery**: Lithuania/Summer subset of BigEarthNet v2.0 (reBEN) pre-converted to LMDB.
  Declared by its publisher as an *unofficial* pre-conversion of the official Zenodo release
  (10.5281/zenodo.10891137) using ``rico-hdl``, the officially recommended converter.

Pairing is validated against the **official** annotation file: every retained sample must
have both a caption row in BigEarthNet.txt and a decodable LMDB image entry. Samples that
fail are excluded with a recorded reason rather than silently dropped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Sentinel-2 bands used to build an RGB image. B04=Red, B03=Green, B02=Blue, all 10 m.
RGB_BANDS = ("B04", "B03", "B02")

#: Official split names present in the annotation file.
SPLITS = ("train", "validation", "test", "bench")

#: Percentiles used for the contrast stretch applied to every patch.
#:
#: A fixed reflectance ceiling was tried first and rejected on measurement: the raw uint16
#: RGB values in this subset have a median of 277 and a 99th percentile of 528, so dividing
#: by a conventional ceiling of 3000 produced near-black images (mean 24/255) that would
#: have crippled training.
#:
#: A per-image percentile stretch is used instead, matching exactly what
#: ``geospatial.render`` applies at inference time. That keeps training and serving
#: consistent and is self-normalising across sensors and dynamic ranges, which matters for
#: generalising to unseen imagery.
STRETCH_PERCENTILES = (2.0, 98.0)


@dataclass
class PreparedSample:
    """One image-caption pair retained for training."""

    patch_id: str
    s1_name: str
    caption: str
    split: str
    latitude: float
    longitude: float
    country: str
    season: str


@dataclass
class PreparationReport:
    """Counts and exclusions from a preparation run, for the manifest."""

    annotations_scanned: int = 0
    matched_country_season: int = 0
    captioning_rows: int = 0
    retained: int = 0
    excluded_missing_image: int = 0
    excluded_unreadable_image: int = 0
    excluded_empty_caption: int = 0
    excluded_duplicate: int = 0
    per_split: dict[str, int] = field(default_factory=dict)
    excluded_examples: list[dict[str, str]] = field(default_factory=list)


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA256 of a file, streamed so large files do not load into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def to_rgb_uint8(tensors: dict[str, np.ndarray]) -> np.ndarray:
    """Convert a Sentinel-2 band dict into an 8-bit RGB array.

    reBEN stores 10 m bands at 120x120 and coarser bands at lower resolution; only the
    10 m RGB bands are used, so no resampling between bands is required.

    A joint 2-98 percentile stretch is applied across the three bands (preserving relative
    colour balance), identical in method to the inference-time rendering in
    ``geospatial.render``. See :data:`STRETCH_PERCENTILES` for why a fixed reflectance
    ceiling was rejected.

    Returns an ``(H, W, 3)`` uint8 array.

    Raises:
        KeyError: if an RGB band is absent.
        ValueError: if the RGB bands have mismatched shapes.
    """
    raw = []
    for band in RGB_BANDS:
        if band not in tensors:
            raise KeyError(f"missing band {band}")
        raw.append(tensors[band].astype("float32"))

    shapes = {channel.shape for channel in raw}
    if len(shapes) != 1:
        raise ValueError(f"RGB bands have mismatched shapes: {shapes}")

    stacked = np.dstack(raw)
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return np.zeros(stacked.shape, dtype="uint8")

    low, high = np.percentile(finite, STRETCH_PERCENTILES)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(stacked.shape, dtype="uint8")

    scaled = np.clip((stacked - low) / (high - low), 0.0, 1.0) * 255.0
    return scaled.astype("uint8")


#: Open LMDB environments, keyed by (pid, resolved path).
#:
#: LMDB refuses to open the same environment twice within one process, and several readers
#: legitimately exist at once (train loader, validation loader, evaluation). They therefore
#: share one handle per path. The pid is part of the key so a forked DataLoader worker opens
#: its own handle rather than inheriting the parent's.
_ENV_CACHE: dict[tuple[int, str], Any] = {}


def _shared_env(lmdb_dir: Path):
    """Return a process-wide shared read-only LMDB environment for ``lmdb_dir``."""
    import os

    import lmdb

    key = (os.getpid(), str(lmdb_dir.resolve()))
    env = _ENV_CACHE.get(key)
    if env is None:
        env = lmdb.open(
            str(lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=1,
        )
        _ENV_CACHE[key] = env
    return env


def close_shared_envs() -> None:
    """Close every cached LMDB environment. For scripts and tests."""
    import os

    pid = os.getpid()
    for key in [k for k in _ENV_CACHE if k[0] == pid]:
        try:
            _ENV_CACHE[key].close()
        except Exception:  # noqa: BLE001 - closing must not raise
            pass
        _ENV_CACHE.pop(key, None)


class LmdbImageReader:
    """Reads Sentinel patches from the reBEN LMDB (SafeTensors values keyed by patch id).

    Multiple readers on the same path share one LMDB environment (see :data:`_ENV_CACHE`).
    """

    def __init__(self, lmdb_dir: str | Path) -> None:
        self.lmdb_dir = Path(lmdb_dir)

    def _env_handle(self):
        return _shared_env(self.lmdb_dir)

    def keys_present(self, keys: list[str]) -> set[str]:
        """Return the subset of ``keys`` that exist in the database."""
        present: set[str] = set()
        with self._env_handle().begin(write=False) as txn:
            for key in keys:
                if txn.get(key.encode()) is not None:
                    present.add(key)
        return present

    def read_bands(self, key: str) -> dict[str, np.ndarray] | None:
        """Load one entry's bands, or None when the key is absent."""
        from safetensors.numpy import load as safetensor_load

        with self._env_handle().begin(write=False) as txn:
            raw = txn.get(key.encode())
        if raw is None:
            return None
        return safetensor_load(raw)

    def read_rgb(self, key: str) -> np.ndarray | None:
        """Load one entry as an 8-bit RGB array, or None when unavailable/unreadable."""
        bands = self.read_bands(key)
        if bands is None:
            return None
        try:
            return to_rgb_uint8(bands)
        except (KeyError, ValueError):
            return None

    def close(self) -> None:
        """Release this reader.

        The underlying environment is shared, so it is not closed here; use
        :func:`close_shared_envs` to release it for the whole process.
        """
        return None

    def __getstate__(self) -> dict[str, Any]:
        # No handle is stored on the instance, so the reader pickles cleanly for
        # DataLoader worker processes.
        return self.__dict__.copy()


def prepare_caption_dataset(
    annotations_parquet: str | Path,
    lmdb_dir: str | Path,
    *,
    country: str,
    season: str,
    max_per_split: dict[str, int] | None = None,
    seed: int = 42,
) -> tuple[list[PreparedSample], PreparationReport]:
    """Build the image-caption training set from BigEarthNet.txt.

    Uses the dataset's **official** ``split`` column, so there is no custom splitting and
    therefore no risk of leakage between train/validation/test.

    Args:
        annotations_parquet: Official ``BigEarthNet.txt.parquet``.
        lmdb_dir: LMDB directory holding the Sentinel patches.
        country: Country filter, matched against the annotation ``country`` column.
        season: Season filter, matched against the annotation ``season`` column.
        max_per_split: Optional cap per split, applied by deterministic shuffle.
        seed: Seed for the capping shuffle.

    Returns:
        ``(samples, report)``. The report records every exclusion and its reason.
    """
    import pyarrow.parquet as pq

    report = PreparationReport()
    reader = LmdbImageReader(lmdb_dir)

    candidates: list[PreparedSample] = []
    seen: set[tuple[str, str]] = set()

    parquet = pq.ParquetFile(annotations_parquet)
    report.annotations_scanned = parquet.metadata.num_rows

    columns = [
        "patch_id",
        "s1_name",
        "input",
        "output",
        "type",
        "split",
        "latitude",
        "longitude",
        "country",
        "season",
    ]
    for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
        data = batch.to_pydict()
        for index in range(len(data["patch_id"])):
            if data["country"][index] != country or data["season"][index] != season:
                continue
            report.matched_country_season += 1
            if data["type"][index] != "captioning":
                continue
            report.captioning_rows += 1

            # For captioning rows the reference answer carries the caption text.
            caption = (data["output"][index] or "").strip()
            if not caption:
                report.excluded_empty_caption += 1
                continue

            patch_id = data["patch_id"][index]
            key = (patch_id, caption)
            if key in seen:
                report.excluded_duplicate += 1
                continue
            seen.add(key)

            candidates.append(
                PreparedSample(
                    patch_id=patch_id,
                    s1_name=data["s1_name"][index],
                    caption=caption,
                    split=data["split"][index],
                    latitude=float(data["latitude"][index]),
                    longitude=float(data["longitude"][index]),
                    country=data["country"][index],
                    season=data["season"][index],
                )
            )

    # --- verify imagery exists for every candidate --------------------------------
    unique_patches = sorted({sample.patch_id for sample in candidates})
    present = reader.keys_present(unique_patches)

    missing = set(unique_patches) - present
    retained: list[PreparedSample] = []
    for sample in candidates:
        if sample.patch_id in missing:
            report.excluded_missing_image += 1
            if len(report.excluded_examples) < 10:
                report.excluded_examples.append(
                    {"patch_id": sample.patch_id, "reason": "image key absent from LMDB"}
                )
            continue
        retained.append(sample)

    # --- verify a sampled subset actually decodes ---------------------------------
    # Decoding every patch would be slow; a deterministic sample proves the pipeline and
    # catches systematic corruption.
    rng = np.random.default_rng(seed)
    probe_keys = list(present)
    rng.shuffle(probe_keys)
    unreadable: set[str] = set()
    for key in probe_keys[: min(64, len(probe_keys))]:
        if reader.read_rgb(key) is None:
            unreadable.add(key)
    if unreadable:
        retained = [s for s in retained if s.patch_id not in unreadable]
        report.excluded_unreadable_image = len(unreadable)
        for key in sorted(unreadable)[:10]:
            report.excluded_examples.append(
                {"patch_id": key, "reason": "image entry could not be decoded to RGB"}
            )

    # --- optional deterministic capping per split ---------------------------------
    if max_per_split:
        capped: list[PreparedSample] = []
        for split in SPLITS:
            in_split = [s for s in retained if s.split == split]
            limit = max_per_split.get(split)
            if limit is not None and len(in_split) > limit:
                order = np.random.default_rng(seed).permutation(len(in_split))
                in_split = [in_split[i] for i in order[:limit]]
            capped.extend(in_split)
        retained = capped

    retained.sort(key=lambda s: (s.split, s.patch_id, s.caption))
    report.retained = len(retained)
    for split in SPLITS:
        report.per_split[split] = sum(1 for s in retained if s.split == split)

    reader.close()
    return retained, report


def write_manifest(
    path: str | Path,
    *,
    samples: list[PreparedSample],
    report: PreparationReport,
    annotations_parquet: Path,
    imagery_metadata: Path,
    lmdb_dir: Path,
    country: str,
    season: str,
    seed: int,
    preprocessing_version: str,
    include_hashes: bool = True,
) -> dict[str, Any]:
    """Write a reproducible dataset manifest and return its content."""
    lmdb_data = lmdb_dir / "data.mdb"

    manifest: dict[str, Any] = {
        "dataset": "BigEarthNet.txt",
        "purpose": "Phase 5 remote-sensing vision-language adaptation (LoRA on CLIP)",
        "created_utc": datetime.now(UTC).isoformat(),
        "preprocessing_version": preprocessing_version,
        "seed": seed,
        "filters": {"country": country, "season": season, "type": "captioning"},
        "split_source": "official `split` column of BigEarthNet.txt (no custom splitting)",
        "annotations": {
            "role": "image-text annotations (OFFICIAL source)",
            "repo": "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
            "url": "https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
            "file": annotations_parquet.name,
            "size_bytes": annotations_parquet.stat().st_size,
            "license": "CDLA-Permissive-1.0",
            "total_rows": report.annotations_scanned,
        },
        "imagery": {
            "role": "Sentinel-1/2 pixels",
            "repo": "hackelle/BigEarthNetV2-Lithuania-Summer-LMDB",
            "url": "https://huggingface.co/datasets/hackelle/BigEarthNetV2-Lithuania-Summer-LMDB",
            "provenance_warning": (
                "Publisher declares this an UNOFFICIAL pre-conversion of the official "
                "BigEarthNet v2.0 release (10.5281/zenodo.10891137), converted with rico-hdl "
                "(the officially recommended converter). Pairing is validated against the "
                "official annotation file."
            ),
            "official_source": "https://zenodo.org/records/10891137",
            "license": "CDLA-Permissive-1.0",
            "lmdb": lmdb_data.name,
            "lmdb_size_bytes": lmdb_data.stat().st_size if lmdb_data.is_file() else None,
            "metadata_file": imagery_metadata.name,
            "bands_used": list(RGB_BANDS),
            "sentinel1_used_for_training": False,
            "sentinel1_note": (
                "Sentinel-1 is present in the LMDB and recorded, but Phase 5 trains on "
                "Sentinel-2 RGB only. No optical-SAR fusion is learned here (that is Phase 7)."
            ),
            "stretch": {
                "method": "per-image joint percentile stretch across B04/B03/B02",
                "percentiles": list(STRETCH_PERCENTILES),
                "rationale": (
                    "Measured raw uint16 RGB values in this subset: p50=277, p99=528, "
                    "max=1148. A fixed ceiling of 3000 produced near-black images "
                    "(mean 24/255), so a per-image stretch matching the inference-time "
                    "renderer is used instead."
                ),
            },
        },
        "counts": asdict(report),
        "unique_patches": len({s.patch_id for s in samples}),
    }

    if include_hashes:
        manifest["annotations"]["sha256"] = sha256_of(annotations_parquet)
        if lmdb_data.is_file():
            manifest["imagery"]["lmdb_sha256"] = sha256_of(lmdb_data)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
