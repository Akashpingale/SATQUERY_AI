"""Generate the SatQuery AI demo dataset (deterministic, synthetic, safe).

The demo rasters are **synthetic** - generated here from fixed seeds, not downloaded from any
external source and containing no hidden ISRO/SAC evaluation data. They are realistic enough to
exercise the real raster pipeline (valid GeoTIFFs with CRS/transform/bands/nodata, declared
optical band roles, a positive SAR-like backscatter band, a genuine temporal change, and a
non-overlapping raster for the incompatible-pair demo).

Because they are re-derivable and large-ish, the ``.tif`` files are gitignored; this script and
the committed manifest (``data/manifests/demo_manifest.json``, with SHA256 checksums) are what
make the demo reproducible on a clean checkout.

    python scripts/make_demo_data.py

Regenerating always produces byte-identical files (fixed seeds), so the checksums are stable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "data" / "demo"
MANIFEST = REPO_ROOT / "data" / "manifests" / "demo_manifest.json"

CRS = "EPSG:32643"  # UTM zone 43N (metres) - metric, so ground areas are meaningful
ORIGIN = (500_000.0, 1_500_000.0)
FAR_ORIGIN = (900_000.0, 1_900_000.0)  # non-overlapping footprint for the incompatible demo
RES = 10.0  # 10 m pixels, like Sentinel-2
SIZE = 160


def _grid_profile(bands: int, dtype: str, origin: tuple[float, float], nodata: float | None):
    profile: dict[str, object] = {
        "driver": "GTiff",
        "width": SIZE,
        "height": SIZE,
        "count": bands,
        "dtype": dtype,
        "crs": CRS,
        "transform": from_origin(origin[0], origin[1], RES, RES),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    return profile


def _write(
    path: Path, array: np.ndarray, *, dtype: str, origin=ORIGIN, band_names=None, nodata=None
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **_grid_profile(array.shape[0], dtype, origin, nodata)) as dst:
        dst.write(array.astype(dtype))
        for index, name in enumerate(band_names or [], start=1):
            dst.set_band_description(index, name)


def _optical(seed: int, *, change_block: bool = False) -> np.ndarray:
    """A 3-band optical scene (Red, Green, NIR) with vegetation, a water strip, and texture."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    red = np.clip(700 + 130 * np.sin(cols / 21.0) + rng.normal(0, 28, (SIZE, SIZE)), 1, 9000)
    green = np.clip(950 + rng.normal(0, 28, (SIZE, SIZE)), 1, 9000)
    nir = np.clip(3100 + 220 * np.cos(rows / 24.0) + rng.normal(0, 45, (SIZE, SIZE)), 1, 9000)

    # A water strip: NIR collapses, green stays -> NDWI positive, optically dark.
    water = slice(int(SIZE * 0.72), int(SIZE * 0.86))
    red[water, :] = 480
    green[water, :] = 1100
    nir[water, :] = 300

    if change_block:
        # A bright development-like block appears at T2 (north-west), spectrally raised.
        block = (
            slice(int(SIZE * 0.15), int(SIZE * 0.42)),
            slice(int(SIZE * 0.20), int(SIZE * 0.52)),
        )
        red[block] = 2600
        green[block] = 2500
        nir[block] = 2400

    return np.stack([red, green, nir])


def _sar(seed: int) -> np.ndarray:
    """A single-band, positive, speckled SAR-like backscatter scene on the same grid."""
    rng = np.random.default_rng(seed)
    sar = np.clip(0.20 + rng.gamma(2.0, 0.05, (SIZE, SIZE)), 1e-3, 6.0)
    water = slice(int(SIZE * 0.72), int(SIZE * 0.86))
    sar[water, :] = 0.03  # smooth open water: dark in SAR
    # A rough, bright structure (matches the optical change block position) - SAR-visible.
    block = (slice(int(SIZE * 0.15), int(SIZE * 0.42)), slice(int(SIZE * 0.20), int(SIZE * 0.52)))
    sar[block] = np.clip(1.5 + rng.gamma(2.0, 0.2, sar[block].shape), 0, 6)
    return sar[None, :, :]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


DemoFile = tuple[str, str, dict]


def main() -> int:
    DEMO.mkdir(parents=True, exist_ok=True)

    # Remove any stale demo rasters so the set is exactly what the manifest describes.
    for stale in DEMO.glob("*.tif"):
        stale.unlink()

    names = ["Red", "Green", "NIR"]
    files: list[DemoFile] = []

    _write(DEMO / "demo_optical.tif", _optical(11), dtype="uint16", band_names=names)
    files.append(
        (
            "demo_optical.tif",
            "single optical / multispectral (3-band, declared bands)",
            {"role": "optical", "modality": "optical", "bands": names},
        )
    )

    _write(DEMO / "demo_sar.tif", _sar(12), dtype="float32")
    files.append(
        (
            "demo_sar.tif",
            "single SAR-like backscatter (1-band, positive)",
            {"role": "sar", "modality": "sar", "bands": ["backscatter"]},
        )
    )

    _write(DEMO / "demo_optical_t1.tif", _optical(21), dtype="uint16", band_names=names)
    _write(
        DEMO / "demo_optical_t2.tif",
        _optical(21, change_block=True),
        dtype="uint16",
        band_names=names,
    )
    files.append(
        ("demo_optical_t1.tif", "temporal T1 (earlier)", {"role": "t1", "modality": "optical"})
    )
    files.append(
        (
            "demo_optical_t2.tif",
            "temporal T2 (later; contains a change block)",
            {"role": "t2", "modality": "optical"},
        )
    )

    _write(
        DEMO / "demo_incompatible.tif",
        _optical(31),
        dtype="uint16",
        band_names=names,
        origin=FAR_ORIGIN,
    )
    files.append(
        (
            "demo_incompatible.tif",
            "optical raster with a non-overlapping footprint (for the incompatible-pair demo)",
            {"role": "t2", "modality": "optical"},
        )
    )

    manifest = {
        "dataset": "SatQuery AI demo data",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source": "synthetic (generated by scripts/make_demo_data.py)",
        "license": "CC0-1.0 (synthetic, no third-party content)",
        "attribution": "None required; synthetic data created for this project.",
        "hidden_evaluation_data": False,
        "grid": {"crs": CRS, "origin": list(ORIGIN), "resolution_m": RES, "size_px": SIZE},
        "preprocessing": "None; written directly as GeoTIFF at the grid above.",
        "regenerate": "python scripts/make_demo_data.py (deterministic; fixed seeds)",
        "demonstrations": {
            "single_optical": ["demo_optical.tif"],
            "single_sar": ["demo_sar.tif"],
            "temporal": ["demo_optical_t1.tif", "demo_optical_t2.tif"],
            "optical_sar_fusion": ["demo_optical.tif", "demo_sar.tif"],
            "incompatible_pair": ["demo_optical.tif", "demo_incompatible.tif"],
        },
        "files": [],
    }

    for filename, purpose, meta in files:
        path = DEMO / filename
        manifest["files"].append(
            {
                "file": filename,
                "purpose": purpose,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                **meta,
                "expected_metadata": {
                    "crs": CRS,
                    "width": SIZE,
                    "height": SIZE,
                    "resolution_m": [RES, RES],
                    "georeferenced": True,
                },
            }
        )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(files)} demo rasters to {DEMO}")
    for entry in manifest["files"]:
        print(f"  {entry['file']:26} {entry['size_bytes'] / 1024:7.0f} KB  {entry['sha256'][:12]}…")
    print(f"Manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
