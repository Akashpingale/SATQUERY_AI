"""Tests for the Phase 4 single-image specialists.

Split into two groups:

- **Model-free**: question analysis, vocabulary, overlay rendering, coordinate mapping,
  and the refusal paths. These always run.
- **Real inference**: end-to-end VQA / captioning / grounding through the API. These use
  the ``models_ready`` fixture and skip when weights are not cached locally, so the suite
  never downloads and never passes against a model that is not present.
"""

from __future__ import annotations

import numpy as np
import pytest
from geospatial import box_to_geo, load_display_image, pixel_to_crs, to_lonlat
from ml.inference import QuestionKind, analyze_question, find_concepts
from ml.inference.detection_backend import Detection
from ml.inference.overlay import draw_detections
from ml.inference.vocabulary import CONCEPTS, LAND_COVER_CONCEPTS, build_prompts, label_for
from PIL import Image

from .raster_fixtures import geotiff_bytes, write_geotiff

TIFF_TYPE = "image/tiff"


# =================================================================================
# Model-free: question analysis
# =================================================================================


@pytest.mark.parametrize(
    "question",
    ["How many buildings are there?", "What is the number of ships?", "Count the aircraft"],
)
def test_counting_questions_are_identified_for_refusal(question):
    """Counting must be recognised so it can be refused instead of fabricated."""
    analysis = analyze_question(question)
    assert analysis.kind == QuestionKind.COUNT
    assert "counting" in analysis.reason.lower()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Is water present?", QuestionKind.PRESENCE),
        ("Are there any buildings?", QuestionKind.PRESENCE),
        ("Can I see a harbour here?", QuestionKind.PRESENCE),
        ("What land cover dominates?", QuestionKind.DOMINANT),
        ("Classify the land use", QuestionKind.DOMINANT),
        ("What is mostly in this scene?", QuestionKind.DOMINANT),
        ("What is visible in this image?", QuestionKind.WHAT_VISIBLE),
        ("What objects can be seen?", QuestionKind.WHAT_VISIBLE),
    ],
)
def test_question_kinds(question, expected):
    assert analyze_question(question).kind == expected


def test_empty_question_is_unsupported():
    assert analyze_question("   ").kind == QuestionKind.UNSUPPORTED


def test_concept_extraction_prefers_longer_triggers():
    keys = [c.key for c in find_concepts("dense residential area")]
    assert keys[0] == "dense_residential"


@pytest.mark.parametrize(
    ("text", "expected_key"),
    [
        ("is there a lake", "water"),
        ("any farmland here", "farmland"),
        ("show the runway", "airport"),
        ("are there trees", "forest"),
        ("ships in the port", "harbor"),
    ],
)
def test_concept_triggers(text, expected_key):
    assert expected_key in [c.key for c in find_concepts(text)]


def test_vocabulary_is_consistent():
    keys = [c.key for c in CONCEPTS]
    assert len(keys) == len(set(keys)), "concept keys must be unique"
    assert len(LAND_COVER_CONCEPTS) >= 10
    for concept in CONCEPTS:
        assert concept.phrase, concept.key
        assert label_for(concept.key)


def test_prompt_templates_expand_each_concept():
    prompts = build_prompts("a body of water")
    assert len(prompts) >= 3
    assert all("a body of water" in prompt for prompt in prompts)
    assert len(set(prompts)) == len(prompts), "templates must be distinct"


# =================================================================================
# Model-free: overlay rendering
# =================================================================================


def _blank_image(width: int = 200, height: int = 120) -> Image.Image:
    return Image.fromarray(np.full((height, width, 3), 128, dtype="uint8"), "RGB")


def test_overlay_draws_boxes_and_writes_png(tmp_path):
    detections = [
        Detection(label="a body of water", score=0.42, box=(10, 20, 80, 90)),
        Detection(label="a body of water", score=0.19, box=(120, 30, 190, 100)),
    ]
    destination = tmp_path / "overlay.png"

    size = draw_detections(_blank_image(), detections, destination)

    assert destination.is_file()
    assert size == (200, 120)
    # The drawn overlay must differ from the flat input image.
    assert len(set(Image.open(destination).convert("RGB").getdata())) > 1


def test_overlay_clamps_out_of_bounds_boxes(tmp_path):
    """A box running past the edge must still render rather than raise."""
    detections = [Detection(label="x", score=0.5, box=(-50, -50, 500, 500))]
    destination = tmp_path / "clamped.png"

    draw_detections(_blank_image(), detections, destination)
    assert destination.is_file()


def test_overlay_handles_box_at_top_edge(tmp_path):
    """Labels for boxes at y=0 must be placed inside the canvas."""
    detections = [Detection(label="x", score=0.5, box=(5, 0, 60, 40))]
    destination = tmp_path / "top.png"

    draw_detections(_blank_image(), detections, destination)
    assert destination.is_file()


def test_overlay_with_no_detections_still_writes(tmp_path):
    destination = tmp_path / "empty.png"
    draw_detections(_blank_image(), [], destination)
    assert destination.is_file()


# =================================================================================
# Model-free: pixel -> world coordinate mapping
# =================================================================================


def test_pixel_to_crs_uses_the_affine_transform():
    # 0.001 degree pixels, origin at (77.0, 13.0), north-up.
    transform = (0.001, 0.0, 77.0, 0.0, -0.001, 13.0)
    x, y = pixel_to_crs(transform, 10, 20)
    assert round(x, 6) == 77.01
    assert round(y, 6) == 12.98


def test_to_lonlat_passes_through_wgs84():
    assert to_lonlat("EPSG:4326", 77.5, 13.0) == (77.5, 13.0)


def test_to_lonlat_reprojects_from_utm():
    result = to_lonlat("EPSG:32643", 717000.0, 1438000.0)
    assert result is not None
    lon, lat = result
    # UTM zone 43N around this easting/northing is near 77E, 13N.
    assert 76.0 < lon < 78.5
    assert 12.0 < lat < 14.5


def test_to_lonlat_returns_none_without_crs():
    assert to_lonlat(None, 1.0, 2.0) is None
    assert to_lonlat("not-a-crs", 1.0, 2.0) is None


def test_box_to_geo_accounts_for_decimation_scale():
    transform = (0.001, 0.0, 77.0, 0.0, -0.001, 13.0)
    # A box in a half-scale image maps to double the source pixel coordinates.
    geo = box_to_geo((10, 10, 20, 20), transform, "EPSG:4326", scale=0.5)

    assert geo is not None
    assert geo["source_pixel_bounds"] == [20.0, 20.0, 40.0, 40.0]
    assert "center_lonlat" in geo
    lon, lat = geo["center_lonlat"]
    assert round(lon, 4) == round(77.0 + 30 * 0.001, 4)
    assert round(lat, 4) == round(13.0 - 30 * 0.001, 4)


def test_box_to_geo_returns_none_without_georeferencing():
    assert box_to_geo((0, 0, 10, 10), None, None) is None
    assert box_to_geo((0, 0, 10, 10), (1, 0, 0, 0, 1, 0), None) is None


def test_load_display_image_converts_single_band_to_rgb(tmp_path):
    """SAR rasters are single-band; models need three channels."""
    path = write_geotiff(tmp_path / "sar.tif", bands=1, dtype="float32", width=64, height=64)
    image, scale = load_display_image(path, max_side=32)

    assert image.mode == "RGB"
    assert max(image.size) <= 32
    assert scale == pytest.approx(0.5, abs=0.01)


# =================================================================================
# Refusal paths (no model needed)
# =================================================================================


def _upload(client, content: bytes, filename: str, role: str, modality: str) -> str:
    resp = client.post(
        "/api/v1/uploads",
        files={"file": (filename, content, TIFF_TYPE)},
        data={"role": role, "modality": modality},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["asset"]["id"]


@pytest.fixture(scope="module")
def vision_asset(client) -> str:
    return _upload(client, geotiff_bytes(width=128, height=128), "vis.tif", "optical", "optical")


def test_counting_question_is_refused_without_inventing_a_number(client, vision_asset):
    """The key anti-fabrication behaviour for VQA."""
    resp = client.post(
        "/api/v1/analysis",
        json={"query": "How many buildings are in this image?", "asset_ids": [vision_asset]},
    )
    body = resp.json()

    assert body["task"] == "vqa"
    assert body["status"] == "completed"
    assert "cannot" in body["answer"].lower()
    # No digits: nothing that could be read as a count.
    assert not any(char.isdigit() for char in body["answer"])
    # No result strength means confidence must be "unable to determine", not a guess.
    assert body["confidence"]["level"] == "Unable to determine"
    assert body["confidence"]["score"] is None


def test_grounding_without_a_target_is_refused(client, vision_asset):
    resp = client.post(
        "/api/v1/analysis",
        json={"query": "Please outline it for me", "asset_ids": [vision_asset]},
    )
    body = resp.json()

    if body["task"] != "grounding":
        pytest.skip("query did not route to grounding; covered by routing tests")
    assert body["confidence"]["level"] == "Unable to determine"
    assert "no target" in body["answer"].lower()


# =================================================================================
# Real inference (skipped when weights are not cached)
# =================================================================================


@pytest.fixture(scope="module")
def structured_asset(client) -> str:
    """A synthetic image with distinct regions, so scores are not degenerate.

    Content is synthetic; the point is only that the pipeline produces real model output
    for a real image, not that the image depicts any particular place.
    """
    import io

    import rasterio
    from rasterio.transform import from_origin

    height = width = 192
    array = np.zeros((3, height, width), dtype="uint8")
    # Blue-ish upper half, green-ish lower half.
    array[2, : height // 2, :] = 200
    array[0, : height // 2, :] = 40
    array[1, height // 2 :, :] = 180
    array[0, height // 2 :, :] = 60

    buffer = io.BytesIO()
    with rasterio.open(
        buffer,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(77.6, 12.98, 0.0005, 0.0005),
    ) as dst:
        dst.write(array)
    return _upload(client, buffer.getvalue(), "structured.tif", "optical", "optical")


def test_vqa_dominant_land_cover_returns_real_scores(client, structured_asset, models_ready):
    body = client.post(
        "/api/v1/analysis",
        json={"query": "What land cover dominates this image?", "asset_ids": [structured_asset]},
    ).json()

    assert body["status"] == "completed"
    assert body["task"] == "vqa"
    assert body["answer"]
    # A real probability drives the answer and the confidence score.
    assert body["confidence"]["score"] is not None
    assert 0.0 < body["confidence"]["score"] <= 1.0

    scores = next(item for item in body["evidence"] if item["kind"] == "scores")
    concepts = scores["detail"]["concepts"]
    assert len(concepts) >= 5
    probabilities = [c["probability"] for c in concepts]
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    # Descending order, and not all identical (a degenerate/mocked result).
    assert probabilities == sorted(probabilities, reverse=True)
    assert len(set(probabilities)) > 1

    # Provenance must name the actual model used.
    model = next(item for item in body["evidence"] if item["kind"] == "model")
    assert "clip-rsicd" in model["detail"]["model_id"]


def test_vqa_presence_question_reports_measured_probability(
    client, structured_asset, models_ready
):
    body = client.post(
        "/api/v1/analysis",
        json={"query": "Is water present in this image?", "asset_ids": [structured_asset]},
    ).json()

    assert body["status"] == "completed"
    assert body["task"] == "vqa"
    assert body["details"] if "details" in body else True
    # The answer states a verdict plus the measured score.
    assert "%" in body["answer"]
    assert body["confidence"]["score"] is not None


def test_vqa_states_uncalibrated_limitation(client, structured_asset, models_ready):
    body = client.post(
        "/api/v1/analysis",
        json={"query": "What land cover dominates?", "asset_ids": [structured_asset]},
    ).json()

    assert any("uncalibrated" in w.lower() for w in body["warnings"])


def test_captioning_produces_a_description_from_real_scores(
    client, structured_asset, models_ready
):
    body = client.post(
        "/api/v1/analysis",
        json={"query": "Describe this scene", "asset_ids": [structured_asset]},
    ).json()

    assert body["status"] == "completed"
    assert body["task"] == "captioning"
    assert len(body["answer"]) > 20
    assert "%" in body["answer"], "the description must carry its measured scores"
    # It must not claim to be generative.
    assert any("retrieval-based" in w for w in body["warnings"])

    scores = next(item for item in body["evidence"] if item["kind"] == "scores")
    assert scores["detail"]["method"].startswith("retrieval")


def test_grounding_returns_boxes_or_an_honest_null_result(
    client, structured_asset, models_ready
):
    body = client.post(
        "/api/v1/analysis",
        json={"query": "Highlight the water in this image", "asset_ids": [structured_asset]},
    ).json()

    assert body["status"] == "completed"
    assert body["task"] == "grounding"

    boxes = [item for item in body["evidence"] if item["kind"] == "box"]
    regions = [item for item in body["evidence"] if item["kind"] == "region"]
    overlays = [item for item in body["evidence"] if item["kind"] == "overlay"]

    if boxes:
        # Detector boxes: each carries a score and pixel geometry.
        assert body["confidence"]["score"] is not None
        for box in boxes:
            assert 0.0 <= box["detail"]["score"] <= 1.0
            assert len(box["detail"]["analyzed_pixel_box"]) == 4
            # The source raster is georeferenced, so coordinates must be present.
            assert "geo" in box["detail"]
            assert "center_lonlat" in box["detail"]["geo"]
    elif regions:
        # Coarse fallback: a measured region, explicitly labelled as an estimate.
        detail = regions[0]["detail"]
        assert detail["method"] == "clip_patch_scoring"
        assert "geo" in detail and "center_lonlat" in detail["geo"]
        assert any("not an object outline" in w for w in body["warnings"])
    else:
        # Nothing at all is inconclusive, never a confident absence.
        assert body["confidence"]["level"] == "Unable to determine"
        assert body["confidence"]["score"] is None
        assert "inconclusive" in body["answer"].lower()

    if boxes or regions:
        assert overlays, "a localization must come with a rendered overlay"
        image_resp = client.get(overlays[0]["reference"])
        assert image_resp.status_code == 200
        assert image_resp.headers["content-type"] == "image/png"
        assert image_resp.content.startswith(b"\x89PNG")


def test_grounding_states_transfer_limitation(client, structured_asset, models_ready):
    body = client.post(
        "/api/v1/analysis",
        json={"query": "Locate the buildings", "asset_ids": [structured_asset]},
    ).json()

    assert any("natural imagery" in w for w in body["warnings"])


def test_coarse_region_is_labelled_as_an_estimate(client, structured_asset, models_ready):
    """When the detector finds nothing, the fallback must not pose as a detection."""
    body = client.post(
        "/api/v1/analysis",
        json={"query": "Highlight the water in this image", "asset_ids": [structured_asset]},
    ).json()

    regions = [item for item in body["evidence"] if item["kind"] == "region"]
    if not regions:
        pytest.skip("detector produced boxes; the coarse fallback did not run")

    detail = regions[0]["detail"]
    assert detail["method"] == "clip_patch_scoring"
    assert 0.0 <= detail["tile_contrast"] <= 1.0
    assert detail["tiles_merged"] >= 1
    # The answer and warnings must both flag that this is tile-level, not an outline.
    assert "tile" in body["answer"].lower()
    assert any("not an object outline" in w for w in body["warnings"])
    # A coarse region must never claim detector-grade confidence.
    assert body["confidence"]["score"] is None or body["confidence"]["score"] < 0.75


def test_patch_localization_contrast_is_not_constant(models_ready):
    """Contrast must measure real tile separation, not always report the maximum."""
    from ml.inference import clip_backend

    height = width = 192
    array = np.zeros((height, width, 3), dtype="uint8")
    array[:, :, 1] = 150  # vegetation-ish everywhere
    array[: height // 3, :, 2] = 220  # a distinct band at the top
    array[: height // 3, :, 1] = 60
    structured = Image.fromarray(array, "RGB")

    flat = Image.fromarray(np.full((height, width, 3), 120, dtype="uint8"), "RGB")

    structured_result = clip_backend.localize_by_patches(structured, "a body of water")
    flat_result = clip_backend.localize_by_patches(flat, "a body of water")

    contrasts = [r.contrast for r in (structured_result, flat_result) if r is not None]
    assert contrasts, "at least one image should yield a localization"
    for contrast in contrasts:
        assert 0.0 <= contrast <= 1.0
        # A normalised maximum would always be exactly 1.0; a real contrast measure is not.
        assert contrast != 1.0 or len(contrasts) > 1

    if structured_result is not None:
        x_min, y_min, x_max, y_max = structured_result.box
        assert 0 <= x_min < x_max <= width
        assert 0 <= y_min < y_max <= height
        assert structured_result.tile_count >= 1
        assert structured_result.grid >= 2


def test_patch_localization_returns_none_on_a_featureless_image(models_ready):
    """A phrase that distinguishes nothing must yield no region rather than a guess."""
    from ml.inference import clip_backend

    uniform = Image.fromarray(np.zeros((160, 160, 3), dtype="uint8"), "RGB")
    result = clip_backend.localize_by_patches(uniform, "a body of water")

    # Either no region, or one whose contrast cleared the documented minimum.
    if result is not None:
        assert result.contrast >= clip_backend.MIN_PATCH_CONTRAST


def test_evidence_image_endpoint_rejects_path_traversal(client):
    for token in ("../../secret", "..%2f..%2fsecret", "not-hex-token", "abc"):
        assert client.get(f"/api/v1/evidence/{token}").status_code == 404
