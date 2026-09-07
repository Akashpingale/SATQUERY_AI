"""Identifying which band is which - only when the raster actually says so.

Spectral indices such as NDVI need specific bands. A GeoTIFF's band *order* is not a
standard: assuming "band 4 is near-infrared" is right for some products and wrong for
others, and a wrong assumption produces a confident, plausible, false NDVI.

So this module only reports a band role when the raster declares it, via a GDAL band
description or a metadata tag. When nothing declares it, the role is ``None`` and the
caller must report the index as unavailable rather than guess. That is the whole point of
the module.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import RasterMetadata

#: Tokens that identify a band role. Matched against band descriptions and tags.
#:
#: Sentinel-2 and Landsat band codes are included because they are unambiguous product
#: identifiers, not positional guesses: ``B04`` means red in every Sentinel-2 product.
_ROLE_TOKENS: dict[str, tuple[str, ...]] = {
    "blue": ("blue", "b02", "b2", "sr_b2", "band_blue"),
    "green": ("green", "b03", "b3", "sr_b3", "band_green"),
    "red": ("red", "b04", "b4", "sr_b4", "band_red"),
    "nir": (
        "nir",
        "near infrared",
        "near-infrared",
        "b08",
        "b8",
        "b8a",
        "sr_b5",
        "band_nir",
    ),
    "swir": ("swir", "shortwave infrared", "b11", "b12", "sr_b6", "sr_b7", "band_swir"),
}

# "red" is a substring of nothing problematic, but "b4" would match inside "b40". Roles are
# therefore matched against whole tokens split on non-alphanumerics, except for the
# multi-word phrases which are matched as substrings.
_PHRASE_TOKENS = frozenset(
    {"near infrared", "near-infrared", "shortwave infrared"}
)


@dataclass(frozen=True)
class BandRoles:
    """Resolved 1-based band indexes for the roles a raster declares."""

    blue: int | None = None
    green: int | None = None
    red: int | None = None
    nir: int | None = None
    swir: int | None = None
    #: How the roles were established, for the execution trace and evidence.
    source: str = "undeclared"

    @property
    def ndvi_available(self) -> bool:
        """NDVI needs red and near-infrared."""
        return self.red is not None and self.nir is not None

    @property
    def ndwi_available(self) -> bool:
        """NDWI (McFeeters) needs green and near-infrared."""
        return self.green is not None and self.nir is not None

    def as_dict(self) -> dict[str, int | None | str]:
        return {
            "blue": self.blue,
            "green": self.green,
            "red": self.red,
            "nir": self.nir,
            "swir": self.swir,
            "source": self.source,
        }


def _tokenize(text: str) -> set[str]:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return {token for token in cleaned.split() if token}


def _role_for(label: str | None) -> str | None:
    """Match a band label to a role, or None when it declares nothing recognisable."""
    if not label:
        return None
    lowered = label.lower()
    tokens = _tokenize(label)
    for role, candidates in _ROLE_TOKENS.items():
        for candidate in candidates:
            if candidate in _PHRASE_TOKENS:
                if candidate in lowered:
                    return role
            elif candidate in tokens:
                return role
    return None


def resolve_band_roles(metadata: RasterMetadata) -> BandRoles:
    """Resolve band roles from what the raster declares.

    Checks GDAL band descriptions first, then metadata tags of the form
    ``band_1_name`` / ``band_1`` / ``B1``. Returns a :class:`BandRoles` with every
    undeclared role set to ``None``.
    """
    found: dict[str, int] = {}
    sources: list[str] = []

    for band in metadata.bands:
        role = _role_for(band.description)
        if role and role not in found:
            found[role] = band.index
            if "band descriptions" not in sources:
                sources.append("band descriptions")

    if len(found) < len(_ROLE_TOKENS):
        lowered_tags = {str(key).lower(): str(value) for key, value in metadata.tags.items()}
        for index in range(1, metadata.band_count + 1):
            for key in (f"band_{index}_name", f"band_{index}", f"b{index}_name"):
                value = lowered_tags.get(key)
                role = _role_for(value)
                if role and role not in found:
                    found[role] = index
                    if "metadata tags" not in sources:
                        sources.append("metadata tags")

    return BandRoles(
        blue=found.get("blue"),
        green=found.get("green"),
        red=found.get("red"),
        nir=found.get("nir"),
        swir=found.get("swir"),
        source=" + ".join(sources) if sources else "undeclared",
    )


def normalized_difference(
    first, second, *, epsilon: float = 1e-6
):
    """``(first - second) / (first + second)``, guarded against divide-by-zero.

    The shared form behind NDVI, NDWI, and NDBI. Returns a float32 array; pixels where the
    denominator is ~0 yield 0.0 rather than NaN or infinity.
    """
    import numpy as np

    a = first.astype("float32", copy=False)
    b = second.astype("float32", copy=False)
    denominator = a + b
    safe = np.abs(denominator) > epsilon
    result = np.zeros(a.shape, dtype="float32")
    np.divide(a - b, denominator, out=result, where=safe)
    return result
