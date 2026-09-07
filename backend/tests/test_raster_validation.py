"""Validation and security tests, including negative/hostile inputs."""

from __future__ import annotations

from geospatial import validate_extension, validate_raster_file

from app.security.files import (
    UnsafePathError,
    build_storage_name,
    is_allowed_content_type,
    resolve_within,
    sanitize_filename,
)

from .raster_fixtures import write_corrupt_tiff, write_geotiff, write_plain_png

# --- format validation ----------------------------------------------------------


def test_supported_extensions_accepted():
    for name in ("scene.tif", "scene.TIFF", "scene.geotiff", "bench.png", "bench.jpg"):
        assert validate_extension(name).valid is True, name


def test_unsupported_extension_rejected_with_reason():
    result = validate_extension("payload.exe")
    assert result.valid is False
    assert result.errors[0].code == "unsupported_format"
    # The message must explain what IS supported, in plain language.
    assert "supported formats" in result.errors[0].message.lower()


def test_missing_extension_rejected():
    result = validate_extension("noextension")
    assert result.valid is False
    assert result.errors[0].code == "missing_extension"


# --- file validation ------------------------------------------------------------


def test_valid_geotiff_passes_with_metadata(tmp_path):
    result = validate_raster_file(write_geotiff(tmp_path / "ok.tif"))
    assert result.valid is True
    assert result.metadata is not None
    assert result.metadata.band_count == 3
    assert result.warnings == []


def test_corrupt_tiff_rejected(tmp_path):
    result = validate_raster_file(write_corrupt_tiff(tmp_path / "bad.tif"))
    assert result.valid is False
    assert result.errors[0].code == "unreadable_raster"


def test_empty_file_rejected(tmp_path):
    empty = tmp_path / "empty.tif"
    empty.write_bytes(b"")
    result = validate_raster_file(empty)
    assert result.valid is False
    assert result.errors[0].code == "empty_file"


def test_missing_file_rejected(tmp_path):
    result = validate_raster_file(tmp_path / "absent.tif")
    assert result.valid is False
    assert result.errors[0].code == "not_found"


def test_extension_mismatch_is_caught_by_content_check(tmp_path):
    """A non-raster renamed to .tif must still be rejected."""
    fake = tmp_path / "disguised.tif"
    fake.write_bytes(b"just some text, definitely not a raster")
    result = validate_raster_file(fake)
    assert result.valid is False
    assert result.errors[0].code == "unreadable_raster"


def test_plain_png_valid_but_warns_about_georeferencing(tmp_path):
    result = validate_raster_file(write_plain_png(tmp_path / "b.png"))
    assert result.valid is True
    assert "not_georeferenced" in [w.code for w in result.warnings]


# --- filename sanitization / path traversal -------------------------------------


def test_sanitize_strips_directory_traversal():
    assert sanitize_filename("../../../etc/passwd.tif") == "passwd.tif"
    assert sanitize_filename("..\\..\\windows\\system32\\evil.tif") == "evil.tif"
    assert sanitize_filename("/absolute/path/scene.tif") == "scene.tif"


def test_sanitize_removes_unsafe_characters_and_nulls():
    # Unsafe runs collapse to '_' and any trailing separator is trimmed.
    assert sanitize_filename("we ird;name&$.tif") == "we_ird_name.tif"
    assert sanitize_filename("a b&c.tif") == "a_b_c.tif"
    assert "\x00" not in sanitize_filename("bad\x00name.tif")


def test_sanitize_drops_unknown_extension():
    """Dangerous extensions must not survive sanitization."""
    assert sanitize_filename("malware.exe") == "malware"
    assert sanitize_filename("script.tif.sh") == "script.tif"


def test_sanitize_handles_empty_and_dot_names():
    assert sanitize_filename("") == "upload"
    assert sanitize_filename("...") == "upload"


def test_build_storage_name_is_unique_and_keeps_extension():
    a = build_storage_name("scene.tif")
    b = build_storage_name("scene.tif")
    assert a != b
    assert a.endswith("_scene.tif")


def test_resolve_within_blocks_escape(tmp_path):
    root = tmp_path / "storage"
    root.mkdir()

    inside = resolve_within(root, "uploads", "file.tif")
    assert str(inside).startswith(str(root.resolve()))

    for attempt in (("..", "outside.tif"), ("uploads", "..", "..", "outside.tif")):
        try:
            resolve_within(root, *attempt)
        except UnsafePathError:
            continue
        raise AssertionError(f"path traversal was not blocked: {attempt}")


def test_content_type_allowlist():
    assert is_allowed_content_type("image/tiff") is True
    assert is_allowed_content_type("image/png") is True
    assert is_allowed_content_type("application/octet-stream") is True
    assert is_allowed_content_type("text/html") is False
    assert is_allowed_content_type("application/x-msdownload") is False
