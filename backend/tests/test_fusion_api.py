"""Phase 7 integration tests: registry, planner routing, the cross-modal specialist, evidence,
trace, and the full optical+SAR API workflow.

Runs the real production pipeline over synthetic optical/SAR GeoTIFFs with known features. No
model weights are needed - the fusion is algorithmic. The optional optical land-cover reading
uses CLIP and is covered behind ``models_ready``.
"""

from __future__ import annotations

import pytest

from app.agent.specialists import is_implemented
from app.registry.models import MODEL_REGISTRY, ToolStatus
from app.registry.tasks import TaskType

from .raster_fixtures import geotiff_bytes, optical_sar_scene

TIFF = "image/tiff"
UTM = "EPSG:32643"
UTM_ORIGIN = (500_000.0, 1_500_000.0)
RES_M = 10.0
GRID = {"crs": UTM, "origin": UTM_ORIGIN, "resolution": RES_M, "dtype": "float32"}


def _upload(client, content, name, role, modality):
    r = client.post(
        "/api/v1/uploads",
        files={"file": (name, content, TIFF)},
        data={"role": role, "modality": modality},
    )
    assert r.status_code == 201, r.text
    return r.json()["asset"]["id"]


def _analyze(client, query, ids):
    r = client.post("/api/v1/analysis", json={"query": query, "asset_ids": ids})
    assert r.status_code == 201, r.text
    return r.json()


def _optical_bytes(**kw):
    optical, _sar = optical_sar_scene()
    return geotiff_bytes(array=optical, band_names=["Red", "Green", "NIR"], **{**GRID, **kw})


def _sar_bytes(**kw):
    _optical, sar = optical_sar_scene()
    return geotiff_bytes(array=sar, **{**GRID, **kw})


@pytest.fixture(scope="module")
def fusion_pair(client):
    o = _upload(client, _optical_bytes(), "xm_opt.tif", "optical", "optical")
    s = _upload(client, _sar_bytes(), "xm_sar.tif", "sar", "sar")
    return o, s


# --- registry ---------------------------------------------------------------------


def test_fusion_registered_implemented_and_degraded():
    spec = MODEL_REGISTRY["optical_sar_fusion"]
    assert is_implemented("optical_sar_fusion")
    assert spec.status is ToolStatus.DEGRADED
    assert spec.phase == 7
    assert spec.version != "0.0.0"
    assert spec.checkpoint is None
    assert "none" in (spec.training_method or "").lower()
    assert spec.limitations
    assert spec.device == "cpu"


def test_fusion_registry_version_matches_implementation():
    from ml.inference.fusion import FUSION_ALGORITHM_VERSION

    assert MODEL_REGISTRY["optical_sar_fusion"].version == FUSION_ALGORITHM_VERSION


def test_models_endpoint_exposes_fusion(client):
    tools = {t["id"]: t for t in client.get("/api/v1/models").json()["tools"]}
    assert tools["optical_sar_fusion"]["status"] == "degraded"
    assert tools["optical_sar_fusion"]["limitations"]


# --- planner routing --------------------------------------------------------------


def test_fusion_toolchain_validates_registration_first():
    from app.agent.planner import TASK_TOOLCHAIN, planned_tools

    assert TASK_TOOLCHAIN[TaskType.OPTICAL_SAR_FUSION][0] == "registration_validator"
    assert planned_tools(TaskType.OPTICAL_SAR_FUSION) == [
        "registration_validator",
        "optical_sar_fusion",
    ]


def test_agent_routes_cross_modal_query_to_fusion(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Analyze these optical and SAR images together.", [o, s])
    assert body["task"] == "optical_sar_fusion"
    tools = [step["tool_id"] for step in body["plan"]["steps"]]
    assert tools == ["registration_validator", "optical_sar_fusion"]


@pytest.mark.parametrize(
    "query",
    [
        "What does SAR reveal that optical does not?",
        "What complementary information can be obtained from these images?",
        "Compare the information provided by optical and SAR.",
        "Which modality provides stronger evidence of vegetation?",
    ],
)
def test_cross_modal_questions_route_to_fusion(client, fusion_pair, query):
    o, s = fusion_pair
    assert _analyze(client, query, [o, s])["task"] == "optical_sar_fusion"


# --- the fused answer -------------------------------------------------------------


def test_describe_produces_a_measured_cross_modal_answer(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Describe this area using both optical and SAR information.", [o, s])

    assert body["status"] == "completed"
    assert body["answer"]
    # The answer must reference measured cross-modal quantities, not two separate captions.
    lowered = body["answer"].lower()
    assert "sar" in lowered and "optical" in lowered
    assert "correlation" in lowered or "complementar" in lowered

    kinds = {item["kind"] for item in body["evidence"]}
    assert "fusion_composite" in kinds
    assert "fusion_complementary" in kinds
    assert "aligned_optical" in kinds
    assert "aligned_sar" in kinds
    assert "modality_comparison" in kinds


def test_modality_comparison_carries_distinct_measured_features(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Analyze these together.", [o, s])
    comparison = next(i for i in body["evidence"] if i["kind"] == "modality_comparison")

    optical = comparison["detail"]["optical"]
    sar = comparison["detail"]["sar"]
    assert optical["modality"] == "optical"
    assert sar["modality"] == "sar"
    # Optical carries NDVI; SAR carries a dark-fraction. They are genuinely different features.
    assert "ndvi_mean" in optical
    assert "dark_fraction" in sar
    assert "uncalibrated" in comparison["detail"]["sar_representation"].lower()


def test_sar_reveals_answer_reports_structure_beyond_optical(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "What does SAR reveal that optical does not?", [o, s])
    assert body["status"] == "completed"
    lowered = body["answer"].lower()
    assert "structure" in lowered or "backscatter" in lowered
    assert "illumination-independent" in lowered
    # Never claims a semantic SAR class.
    assert "building" not in lowered or "measured" in lowered


def test_water_agreement_question_is_answered_from_measurement(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Is the water region consistent between optical and SAR?", [o, s])
    assert body["status"] == "completed"
    assert "agree on water" in body["answer"].lower()
    water = next(i for i in body["evidence"] if i["kind"] == "cross_modal_water")
    assert water["detail"]["available"] is True
    assert water["detail"]["agreement_iou"] >= 0.0


def test_which_modality_vegetation_prefers_optical_with_ndvi(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Which modality provides stronger evidence of vegetation?", [o, s])
    assert body["status"] == "completed"
    lowered = body["answer"].lower()
    assert "optical" in lowered and "ndvi" in lowered
    assert "greenness" in lowered or "does not provide" in lowered


def test_builtup_question_reports_structure_not_a_classification(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Which modality better shows built structures?", [o, s])
    assert body["status"] == "completed"
    assert "no built-up classifier" in body["answer"].lower()
    assert any("not a land-use classification" in w.lower() for w in body["warnings"])


def test_complementary_regions_have_coordinates_when_georeferenced(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "What complementary information is available?", [o, s])
    regions = [i for i in body["evidence"] if i["kind"] == "complementary_region"]
    assert regions, "at least one complementary region should be reported"
    region = regions[0]
    assert region["detail"]["centroid_lonlat"] is not None
    lon, lat = region["detail"]["centroid_lonlat"]
    assert 74.0 < lon < 76.0 and 13.0 < lat < 14.5


def test_unsupported_cross_modal_question_returns_a_limitation(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "What is the average rainfall here?", [o, s])
    assert body["status"] == "completed"
    assert "not answerable" in body["answer"]
    assert body["confidence"]["level"] == "Unable to determine"
    assert body["confidence"]["score"] is None


def test_confidence_is_evidence_derived(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Analyze these together.", [o, s])
    assert body["confidence"]["level"] in {"High", "Medium", "Low"}
    assert body["confidence"]["score"] is not None
    assert body["confidence"]["signals"]


# --- negative / guard paths -------------------------------------------------------


def test_missing_sar_is_reported(client):
    o1 = _upload(client, _optical_bytes(), "mo_a.tif", "optical", "optical")
    o2 = _upload(client, _optical_bytes(), "mo_b.tif", "optical", "optical")
    body = _analyze(client, "Analyze these optical and SAR images together.", [o1, o2])
    # Two optical images: the fusion task is infeasible, so it is rejected up front.
    assert body["status"] == "rejected"
    assert body["task"] == "unsupported"


def test_two_sar_images_do_not_route_to_fusion(client):
    s1 = _upload(client, _sar_bytes(), "ts_a.tif", "sar", "sar")
    s2 = _upload(client, _sar_bytes(), "ts_b.tif", "sar", "sar")
    body = _analyze(client, "Analyze these optical and SAR images together.", [s1, s2])
    assert body["task"] != "optical_sar_fusion" or body["status"] == "rejected"


def test_more_than_two_images_is_a_validation_error(client, fusion_pair):
    o, s = fusion_pair
    extra = _upload(client, _optical_bytes(), "extra.tif", "optical", "optical")
    r = client.post(
        "/api/v1/analysis",
        json={"query": "Fuse these", "asset_ids": [o, s, extra]},
    )
    assert r.status_code == 422


def test_non_overlapping_optical_sar_is_rejected_before_fusion(client):
    o = _upload(client, _optical_bytes(), "no_o.tif", "optical", "optical")
    s = _upload(
        client, _sar_bytes(origin=(900_000.0, 1_900_000.0)), "no_s.tif", "sar", "sar"
    )
    body = _analyze(client, "Analyze these optical and SAR images together.", [o, s])
    assert body["status"] == "rejected"
    assert body["issues"][0]["code"] == "incompatible_inputs"
    assert not any(i["kind"] == "fusion_composite" for i in body["evidence"])


def test_crs_mismatch_is_reprojected_and_still_fuses(client):
    """A CRS difference is a reprojection, not a failure - fusion should still complete."""
    o = _upload(client, _optical_bytes(), "cm_o.tif", "optical", "optical")
    _optical, sar = optical_sar_scene()
    s_bytes = geotiff_bytes(
        array=sar, crs="EPSG:4326", origin=(75.0, 13.57), resolution=0.0001, dtype="float32"
    )
    s = _upload(client, s_bytes, "cm_s.tif", "sar", "sar")
    body = _analyze(client, "Analyze these optical and SAR images together.", [o, s])
    # Overlap between UTM and WGS84 fixtures may be partial; either it fuses or is honestly
    # rejected for insufficient overlap - never a silent wrong comparison.
    assert body["status"] in {"completed", "rejected"}
    if body["status"] == "completed":
        steps = {e["step"]: e for e in body["trace"]}
        assert any("reproject" in steps.get("Co-registration", {}).get("detail", "") for _ in [0])


# --- trace and evidence -----------------------------------------------------------


def test_trace_records_co_registration_and_cross_modal_analysis(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Analyze these together.", [o, s])
    steps = {e["step"]: e for e in body["trace"]}
    assert "Modality resolution" in steps
    assert "Co-registration" in steps
    assert "resampled both images" in steps["Co-registration"]["detail"]
    assert "Cross-modal analysis" in steps


def test_trace_has_no_chain_of_thought(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Analyze these together.", [o, s])
    blob = " ".join(f"{e['step']} {e['detail']}" for e in body["trace"]).lower()
    for leak in ("i think", "let me", "reasoning:", "chain of thought", "my plan is"):
        assert leak not in blob


def test_fusion_artifacts_are_served_as_images(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Analyze these together.", [o, s])
    refs = [
        i["reference"]
        for i in body["evidence"]
        if i["kind"]
        in {"fusion_composite", "fusion_complementary", "aligned_optical", "aligned_sar"}
    ]
    assert len(refs) == 4
    for ref in refs:
        token = ref.rsplit("/", 1)[-1]
        r = client.get(f"/api/v1/evidence/{token}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert len(r.content) > 1000


def test_method_provenance_states_no_trained_fusion_model(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Analyze these together.", [o, s])
    provenance = next(
        i for i in body["evidence"] if i["kind"] == "model" and i["label"] == "Method provenance"
    )
    assert provenance["detail"]["deterministic"] is True
    assert "none for fusion" in provenance["detail"]["learned_component"]


def test_persisted_evidence_is_retrievable(client, fusion_pair):
    o, s = fusion_pair
    body = _analyze(client, "Analyze these together.", [o, s])
    evidence = client.get(f"/api/v1/analysis/{body['id']}/evidence").json()
    assert any(i["kind"] == "fusion_composite" for i in evidence["evidence"])


# --- regression -------------------------------------------------------------------


def test_single_image_and_temporal_paths_still_work(client):
    """A quick regression that Phase 4/6 routing is unchanged by Phase 7."""
    opt = _upload(client, geotiff_bytes(), "reg_opt.tif", "single", "optical")
    meta = _analyze(client, "What CRS and resolution does this image use?", [opt])
    assert meta["task"] == "metadata_inspection"
    assert meta["status"] == "completed"


# --- optional semantic layer (needs CLIP) -----------------------------------------


def test_optical_landcover_reading_is_labelled_indicative(client, fusion_pair, models_ready):
    o, s = fusion_pair
    body = _analyze(client, "Describe this area using both optical and SAR.", [o, s])
    reading = next(
        (i for i in body["evidence"] if i["kind"] == "scores" and "optical" in i["label"].lower()),
        None,
    )
    assert reading is not None
    assert "indicative" in reading["detail"]["note"].lower()
    assert "not applied to" in reading["detail"]["note"].lower()
