"""Model/tool registry.

Each entry declares a specialist the agent may select. In Phase 1 every specialist is
``planned`` with truthful metadata (no checkpoint, license "TBD" until chosen). This
enforces from day one the rule that the agent never invents tools, parameters, or
model names. Statuses flip to ``available`` as real implementations land.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from .tasks import Modality, TaskType


class ToolStatus(StrEnum):
    AVAILABLE = "available"  # real implementation + checkpoint present and healthy
    DEGRADED = "degraded"  # usable but limited (e.g. CPU baseline, missing accel)
    PLANNED = "planned"  # declared, not yet implemented
    UNAVAILABLE = "unavailable"  # implemented but currently not loadable


class ParamSpec(BaseModel):
    """Allowlisted parameter definition. The agent may only set declared parameters."""

    name: str
    type: str  # "int" | "float" | "bool" | "str" | "enum"
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    choices: list[str] | None = None
    description: str = ""


class ModelVariant(StrEnum):
    """Whether an entry is the stock model or an adapted one.

    Kept explicit so a user, a judge, or a log reader can always tell which of the two
    produced a given answer.
    """

    BASELINE = "baseline"
    ADAPTED = "adapted"
    NONE = "none"  # support tools that are not models


class ToolSpec(BaseModel):
    """A registered specialist tool/model."""

    id: str
    name: str
    version: str
    task: TaskType
    modalities: list[Modality]
    input_contract: str
    output_contract: str
    allowed_parameters: list[ParamSpec]
    device: str  # "cpu" | "cuda" | "any"
    checkpoint: str | None
    license: str
    status: ToolStatus
    phase: int

    # --- adaptation provenance (Phase 5) -----------------------------------------
    variant: ModelVariant = ModelVariant.NONE
    #: Base model an adapter was trained on top of.
    base_model: str | None = None
    #: Path to the learned adapter, relative to the repository root.
    adapter_path: str | None = None
    #: Dataset the adaptation was trained on.
    training_dataset: str | None = None
    #: How the model was adapted (e.g. "LoRA (contrastive)").
    training_method: str | None = None
    #: Honest benchmark state. Never claim "benchmarked" without a real evaluation.
    benchmark_status: str = "not benchmarked"
    #: Known limitations, surfaced to users alongside results.
    limitations: list[str] = []


def _tok(n: int = 64, hi: int = 256) -> ParamSpec:
    return ParamSpec(
        name="max_new_tokens",
        type="int",
        default=n,
        minimum=1,
        maximum=hi,
        description="Maximum number of generated tokens.",
    )


def _change_algorithm_version() -> str:
    """Version of the Phase 6 change detector, read from the implementation.

    Imported lazily and defensively so the registry - which the API serves on every
    ``/models`` call - stays importable when the ML extras are absent.
    """
    try:
        from ml.inference.temporal import ALGORITHM_VERSION

        return ALGORITHM_VERSION
    except Exception:  # noqa: BLE001 - registry must never fail to import
        return "unknown"


_CHANGE_ALGORITHM_VERSION = _change_algorithm_version()


def _fusion_algorithm_version() -> str:
    """Version of the Phase 7 cross-modal analysis, read from the implementation."""
    try:
        from ml.inference.fusion import FUSION_ALGORITHM_VERSION

        return FUSION_ALGORITHM_VERSION
    except Exception:  # noqa: BLE001 - registry must never fail to import
        return "unknown"


_FUSION_ALGORITHM_VERSION = _fusion_algorithm_version()

#: Allowlisted tuning for the change detector. Every one of these is a real threshold in
#: ``ml.inference.temporal.ChangeConfig``; the ranges keep an agent-proposed value inside
#: territory where the method still behaves.
_CHANGE_PARAMETERS = [
    ParamSpec(
        name="noise_sigma",
        type="float",
        default=3.0,
        minimum=1.0,
        maximum=6.0,
        description=(
            "Robust sigma multiplier setting the noise floor a pixel must clear to count as "
            "changed. Higher is more conservative."
        ),
    ),
    ParamSpec(
        name="min_region_pixels",
        type="int",
        default=25,
        minimum=4,
        maximum=5000,
        description="Smallest connected changed region kept; smaller ones are treated as noise.",
    ),
    ParamSpec(
        name="min_changed_fraction",
        type="float",
        default=0.005,
        minimum=0.0005,
        maximum=0.5,
        description=(
            "Share of the shared area that must change before a change claim is reported."
        ),
    ),
]


MODEL_REGISTRY: dict[str, ToolSpec] = {
    # Phase 4 baselines. Status is DEGRADED rather than AVAILABLE on purpose: these run
    # real inference and return real scores, but they are zero-shot, uncalibrated, and not
    # yet adapted with LoRA on remote-sensing data (Phase 5). Overstating them as fully
    # available would misrepresent their accuracy.
    "vqa": ToolSpec(
        id="vqa",
        name="VQA (baseline)",
        version="clip-rsicd-v2",
        task=TaskType.VQA,
        modalities=[Modality.OPTICAL, Modality.SAR],
        input_contract="{ image_id: str, question: str }",
        output_contract="{ answer: str, score: float, evidence: Evidence[] }",
        allowed_parameters=[],
        device="cpu",
        checkpoint="flax-community/clip-rsicd-v2",
        license="See model card (CLIP fine-tune on RSICD); no license declared upstream",
        status=ToolStatus.DEGRADED,
        phase=4,
        variant=ModelVariant.BASELINE,
        base_model="flax-community/clip-rsicd-v2",
        training_dataset="RSICD (upstream fine-tune); no adaptation by this project",
        training_method="none (zero-shot use of an upstream checkpoint)",
        benchmark_status="not benchmarked",
        limitations=[
            "Zero-shot and uncalibrated; scores are relative to the concept vocabulary.",
            "No counting or spatial reasoning.",
        ],
    ),
    # Phase 5: the same architecture with a LoRA adapter this project trained on
    # BigEarthNet.txt. Kept as a separate registry entry so baseline and adapted results are
    # never confused.
    "vqa_adapted": ToolSpec(
        id="vqa_adapted",
        name="VQA (adapted, BigEarthNet.txt LoRA)",
        version="clip-rsicd-v2+lora-ben-txt",
        task=TaskType.VQA,
        modalities=[Modality.OPTICAL, Modality.SAR],
        input_contract="{ image_id: str, question: str }",
        output_contract="{ answer: str, score: float, evidence: Evidence[] }",
        allowed_parameters=[],
        device="cuda",
        checkpoint="models/phase5/adapter",
        license=(
            "Adapter: MIT (this project). Base: see model card, no license declared "
            "upstream. Training data: CDLA-Permissive-1.0"
        ),
        status=ToolStatus.DEGRADED,
        phase=5,
        variant=ModelVariant.ADAPTED,
        base_model="flax-community/clip-rsicd-v2",
        adapter_path="models/phase5/adapter",
        training_dataset="BigEarthNet.txt (Lithuania/Summer captioning subset)",
        training_method="LoRA, symmetric contrastive (InfoNCE) objective",
        benchmark_status="evaluated on held-out BigEarthNet.txt retrieval; see reports/phase5",
        limitations=[
            "Adapted on a Lithuania/Summer Sentinel-2 subset; benefit on other regions, "
            "seasons, or sensors is not established.",
            "Trained on Sentinel-2 RGB only; no optical-SAR fusion was learned.",
            "Captions were truncated to CLIP's 77-token limit (~11% of samples affected).",
            "Scores remain uncalibrated.",
        ],
    ),
    "captioning": ToolSpec(
        id="captioning",
        name="Captioning (baseline)",
        version="clip-rsicd-v2",
        task=TaskType.CAPTIONING,
        modalities=[Modality.OPTICAL, Modality.SAR],
        input_contract="{ image_id: str }",
        output_contract="{ caption: str, score: float, evidence: Evidence[] }",
        allowed_parameters=[],
        device="cpu",
        checkpoint="flax-community/clip-rsicd-v2",
        license="See model card (CLIP fine-tune on RSICD); no license declared upstream",
        status=ToolStatus.DEGRADED,
        phase=4,
        variant=ModelVariant.BASELINE,
        base_model="flax-community/clip-rsicd-v2",
        training_dataset="RSICD (upstream fine-tune); no adaptation by this project",
        training_method="none (retrieval-based composition from zero-shot scores)",
        benchmark_status="not benchmarked",
        limitations=[
            "Retrieval-composed, not generative.",
            "Zero-shot and uncalibrated.",
        ],
    ),
    "captioning_adapted": ToolSpec(
        id="captioning_adapted",
        name="Captioning (adapted, BigEarthNet.txt LoRA)",
        version="clip-rsicd-v2+lora-ben-txt",
        task=TaskType.CAPTIONING,
        modalities=[Modality.OPTICAL, Modality.SAR],
        input_contract="{ image_id: str }",
        output_contract="{ caption: str, score: float, evidence: Evidence[] }",
        allowed_parameters=[],
        device="cuda",
        checkpoint="models/phase5/adapter",
        license=(
            "Adapter: MIT (this project). Base: see model card, no license declared "
            "upstream. Training data: CDLA-Permissive-1.0"
        ),
        status=ToolStatus.DEGRADED,
        phase=5,
        variant=ModelVariant.ADAPTED,
        base_model="flax-community/clip-rsicd-v2",
        adapter_path="models/phase5/adapter",
        training_dataset="BigEarthNet.txt (Lithuania/Summer captioning subset)",
        training_method="LoRA, symmetric contrastive (InfoNCE) objective",
        benchmark_status="evaluated on held-out BigEarthNet.txt retrieval; see reports/phase5",
        limitations=[
            "Still retrieval-composed, not generative: adaptation improved the embedding "
            "space, it did not add text generation.",
            "Adapted on a Lithuania/Summer Sentinel-2 subset only.",
        ],
    ),
    "grounding": ToolSpec(
        id="grounding",
        name="Grounding",
        version="owlvit-base-patch32",
        task=TaskType.GROUNDING,
        modalities=[Modality.OPTICAL, Modality.SAR],
        input_contract="{ image_id: str, target: str }",
        output_contract="{ boxes: Box[], overlay_ref: str|null, score: float }",
        allowed_parameters=[
            ParamSpec(
                name="score_threshold",
                type="float",
                default=0.08,
                minimum=0.0,
                maximum=1.0,
                description="Minimum detection score to report as a finding.",
            )
        ],
        device="cpu",
        checkpoint="google/owlvit-base-patch32",
        license="Apache-2.0",
        status=ToolStatus.DEGRADED,
        phase=4,
    ),
    # Phase 6. Change detection is algorithmic, not learned: radiometrically normalised
    # change vector analysis with Otsu + noise-floor thresholding and spatial-coherence
    # filtering. It is deterministic and needs no weights or GPU, so `checkpoint` is None
    # because there is nothing to load - not because something is missing.
    #
    # Status is DEGRADED, not AVAILABLE: the method is real and reproducible, but it has not
    # been evaluated against a labelled change-detection benchmark, and it detects
    # radiometric change rather than classifying change *type*.
    "temporal_change": ToolSpec(
        id="temporal_change",
        name="TemporalChange (change vector analysis)",
        version=_CHANGE_ALGORITHM_VERSION,
        task=TaskType.TEMPORAL_CHANGE,
        modalities=[Modality.OPTICAL, Modality.SAR],
        input_contract="{ t1_image_id: str, t2_image_id: str }",
        output_contract=(
            "{ description: str, change_map_ref: str|null, regions: Region[], "
            "indices: IndexChange[], score: float }"
        ),
        allowed_parameters=_CHANGE_PARAMETERS,
        device="cpu",
        checkpoint=None,
        license="OSS (NumPy/SciPy/Rasterio); no third-party checkpoint",
        status=ToolStatus.DEGRADED,
        phase=6,
        variant=ModelVariant.NONE,
        training_dataset="none - the detector is algorithmic, not trained",
        training_method="none (deterministic change vector analysis)",
        benchmark_status="not benchmarked",
        limitations=[
            "Detects radiometric change; it does not classify what kind of change occurred.",
            "Requires both dates in the same modality - optical vs SAR is refused.",
            "NDVI/NDWI are only computed when the raster declares its band roles.",
            "Reports inconclusive when the pair is misregistered by more than one pixel.",
        ],
    ),
    "change_vqa": ToolSpec(
        id="change_vqa",
        name="ChangeVQA (measurement-grounded)",
        version=_CHANGE_ALGORITHM_VERSION,
        task=TaskType.CHANGE_VQA,
        modalities=[Modality.OPTICAL, Modality.SAR],
        input_contract="{ t1_image_id: str, t2_image_id: str, question: str }",
        output_contract="{ answer: str, score: float, evidence: Evidence[] }",
        allowed_parameters=_CHANGE_PARAMETERS,
        device="cpu",
        checkpoint=None,
        license="OSS (NumPy/SciPy/Rasterio); no third-party checkpoint",
        status=ToolStatus.DEGRADED,
        phase=6,
        variant=ModelVariant.NONE,
        training_dataset="none - answers are derived from measurements, not generated",
        training_method="none (deterministic change vector analysis)",
        benchmark_status="not benchmarked; CDVQA evaluation is Phase 9",
        limitations=[
            "Answers only questions the measurement supports: occurrence, location, extent, "
            "and NDVI/NDWI direction. Anything else returns a stated limitation.",
            "No built-up/urban classifier exists, so construction is never confirmed.",
            "Vegetation and water trends need declared band roles; otherwise it declines.",
        ],
    ),
    # Phase 9: a LEARNED change-VQA classifier trained on the real CDVQA dataset (questions)
    # with SECOND imagery. Registered and invocable; benchmarked on the held-out CDVQA test
    # split. DEGRADED, not AVAILABLE, and NOT the default change-VQA route: it is trained on
    # 0.1524 m aerial imagery with a closed 19-answer vocabulary, so the general deterministic
    # detector remains the production default for arbitrary imagery.
    "cdvqa_trained": ToolSpec(
        id="cdvqa_trained",
        name="ChangeVQA (trained classifier, CDVQA)",
        version="cdvqa-clip-rsicd-frozen+head",
        task=TaskType.CHANGE_VQA,
        modalities=[Modality.OPTICAL],
        input_contract="{ t1_image_id: str, t2_image_id: str, question: str }",
        output_contract="{ answer: str, score: float, evidence: Evidence[] }",
        allowed_parameters=[],
        device="cpu",
        checkpoint="models/phase9/cdvqa/cdvqa_best.pt",
        license="Model: MIT (this project). Training data: CDVQA (research use) + SECOND.",
        status=ToolStatus.DEGRADED,
        phase=9,
        variant=ModelVariant.ADAPTED,
        base_model="flax-community/clip-rsicd-v2 (frozen image encoder)",
        adapter_path="models/phase9/cdvqa",
        training_dataset="CDVQA questions + SECOND imagery (train split only)",
        training_method="frozen CLIP features + trained GRU question encoder + fusion classifier",
        benchmark_status="CDVQA held-out Test acc 0.71 / Test2 0.66 (see reports/benchmarks/cdvqa)",
        limitations=[
            "Trained on SECOND 0.1524 m aerial imagery; benefit on other sensors/resolutions "
            "(e.g. Sentinel) is not established.",
            "Closed 19-answer vocabulary from CDVQA; cannot answer outside it.",
            "Weakest on counting/ratio questions (smallest_change ~0.32); confidence is "
            "an uncalibrated softmax.",
            "Not the default route; the deterministic detector is used for general imagery.",
        ],
    ),
    # Phase 9: a LEARNED binary change detector trained on the SECOND dataset (additional public
    # dataset, NOT SIH-provided). Registered/invocable; benchmarked on an internal held-out split
    # (IoU 0.42 / F1 0.59 vs a deterministic reference 0.22 / 0.37). DEGRADED and NOT the default:
    # trained on 0.1524 m aerial RGB, binary change only; the deterministic CVA stays default.
    "second_change_learned": ToolSpec(
        id="second_change_learned",
        name="LearnedChange (Siamese U-Net, SECOND)",
        version="second-siamese-unet-binary",
        task=TaskType.TEMPORAL_CHANGE,
        modalities=[Modality.OPTICAL],
        input_contract="{ t1_image_id: str, t2_image_id: str }",
        output_contract="{ answer: str, score: float, evidence: Evidence[] }",
        allowed_parameters=[],
        device="cpu",
        checkpoint="models/phase9/second/second_best.pt",
        license="Model: MIT (this project). Training data: SECOND (additional public dataset).",
        status=ToolStatus.DEGRADED,
        phase=9,
        variant=ModelVariant.ADAPTED,
        base_model="Siamese U-Net (trained from scratch)",
        adapter_path="models/phase9/second",
        training_dataset="SECOND train split (binary change derived from semantic labels)",
        training_method="supervised segmentation (Dice+BCE), internal train/val/test split",
        benchmark_status="internal held-out IoU 0.42 / F1 0.59 (see reports/benchmarks/second)",
        limitations=[
            "Binary change only (changed vs unchanged); not from-to semantic class.",
            "Trained on SECOND 0.1524 m aerial RGB; generalization to other sensors/resolutions "
            "not established.",
            "Held-out split is an internal split of the provided train set, not the official "
            "SECOND test set.",
            "Not the default route; the deterministic CVA is used for general imagery.",
        ],
    ),
    # Phase 14: an EXTERNAL trained BLIP Change-VQA checkpoint (change_model/), integrated as an
    # ADDITIONAL generative Change-VQA specialist. Real weights, real generation. DEGRADED and
    # NOT the default: it is single-image (a temporal pair is shown under a frozen convention),
    # and on CDVQA held-out it scored Test 0.5467 / Test2 0.5033 - BELOW the project's own
    # cdvqa_trained (0.7114 / 0.6632). Reachable when CHANGE_VQA_MODEL selects it; the
    # deterministic change_vqa stays the production default.
    "change_vqa_blip": ToolSpec(
        id="change_vqa_blip",
        name="ChangeVQA (external trained BLIP, generative)",
        version="blip-vqa-cdvqa-external",
        task=TaskType.CHANGE_VQA,
        modalities=[Modality.OPTICAL],
        input_contract="{ t1_image_id: str, t2_image_id: str, question: str }",
        output_contract="{ answer: str, score: float, evidence: Evidence[] }",
        allowed_parameters=[],
        device="cpu",
        checkpoint="change_model/model.safetensors",
        license="External checkpoint (provided out-of-band); weights unchanged and gitignored.",
        status=ToolStatus.DEGRADED,
        phase=14,
        variant=ModelVariant.NONE,
        base_model="BlipForQuestionAnswering (Salesforce BLIP VQA base), CDVQA fine-tune",
        adapter_path=None,
        training_dataset="CDVQA (external fine-tune; NOT trained by this project)",
        training_method="external BLIP VQA fine-tuning (generative); loaded read-only",
        benchmark_status=(
            "CDVQA held-out Test 0.5467 / Test2 0.5033 (n=600 each; see reports/change_model)"
        ),
        limitations=[
            "Single-image generative VQA; for a T1/T2 pair one image is shown under the "
            "frozen 't1' convention (selected on CDVQA validation only). It is NOT a "
            "two-image temporal change detector.",
            "On CDVQA held-out it scores BELOW the project's cdvqa_trained (0.5467/0.5033 "
            "vs 0.7114/0.6632), so it is an additional option, not a replacement.",
            "Confidence is an uncalibrated mean token probability, not a probability of "
            "correctness.",
            "Not the default route; the deterministic change_vqa stays default. Reachable "
            "via CHANGE_VQA_MODEL=blip (or =auto for classification-style questions).",
        ],
    ),
    # Phase 15: the IMPROVED CDVQA classifier. Same GRU+fusion head as Phase 9, but the image
    # features are CLIP-rsicd (512) concatenated with the external change_model BLIP vision
    # encoder (768) - i.e. change_model used as a complementary frozen representation source.
    # Real training on CDVQA train split; val-selected; frozen Test 0.7156 / Test2 0.6668 vs the
    # Phase 9 baseline 0.7114 / 0.6632. Best learned Change-VQA model; still non-default.
    "cdvqa_transfer": ToolSpec(
        id="cdvqa_transfer",
        name="ChangeVQA (improved, CLIP+change_model BLIP features)",
        version="cdvqa-clip-rsicd+blip-transfer",
        task=TaskType.CHANGE_VQA,
        modalities=[Modality.OPTICAL],
        input_contract="{ t1_image_id: str, t2_image_id: str, question: str }",
        output_contract="{ answer: str, score: float, evidence: Evidence[] }",
        allowed_parameters=[],
        device="cpu",
        checkpoint="models/phase15/cdvqa_clip_blip/cdvqa_transfer_best.pt",
        license="Model: MIT (this project). Training data: CDVQA (research use) + SECOND.",
        status=ToolStatus.DEGRADED,
        phase=15,
        variant=ModelVariant.ADAPTED,
        base_model="CLIP-rsicd (frozen) + change_model BLIP vision (frozen) features",
        adapter_path="models/phase15/cdvqa_clip_blip",
        training_dataset="CDVQA questions + SECOND imagery (train split only)",
        training_method=(
            "trained GRU question encoder + fusion classifier on concatenated frozen "
            "CLIP-rsicd + change_model BLIP vision features (real gradient updates)"
        ),
        benchmark_status=(
            "CDVQA frozen Test 0.7156 / Test2 0.6668 (vs Phase 9 baseline 0.7114 / 0.6632; "
            "see reports/phase15)"
        ),
        limitations=[
            "Closed 19-answer vocabulary from CDVQA; cannot answer outside it.",
            "Trained on SECOND 0.1524 m aerial RGB; benefit on other sensors/resolutions "
            "is not established.",
            "Improvement over the Phase 9 baseline is small (+0.42 pp Test / +0.36 pp Test2); "
            "weakest on smallest_change / change_ratio, like the baseline.",
            "Not the default route; the deterministic detector stays default. Reachable via "
            "CHANGE_VQA_MODEL=cdvqa_transfer.",
        ],
    ),
    # Phase 7. Genuine feature-level optical+SAR cross-modal analysis: the two modalities are
    # co-registered onto one grid, per-modality features are extracted, and cross-modal
    # quantities neither image produces alone (correlation/complementarity, the optical-only /
    # SAR-only / shared information partition, and optical-vs-SAR water agreement) are measured.
    # Algorithmic and CPU-only, so `checkpoint` is None because there is nothing to load.
    #
    # DEGRADED, not AVAILABLE: real and reproducible, but not benchmarked, and SAR is treated as
    # uncalibrated backscatter unless the source metadata declares calibration.
    "optical_sar_fusion": ToolSpec(
        id="optical_sar_fusion",
        name="OpticalSARFusion (cross-modal, feature-level)",
        version=_FUSION_ALGORITHM_VERSION,
        task=TaskType.OPTICAL_SAR_FUSION,
        modalities=[Modality.OPTICAL, Modality.SAR],
        input_contract="{ optical_image_id: str, sar_image_id: str, question: str|null }",
        output_contract=(
            "{ answer: str, evidence: Evidence[], complementarity: float|null, "
            "regions: Region[], score: float }"
        ),
        allowed_parameters=[],
        device="cpu",
        checkpoint=None,
        license="OSS (NumPy/SciPy/Rasterio); no third-party checkpoint",
        status=ToolStatus.DEGRADED,
        phase=7,
        variant=ModelVariant.NONE,
        training_dataset="none - the analysis is algorithmic, not trained",
        training_method="none (deterministic feature-level cross-modal analysis)",
        benchmark_status="not benchmarked",
        limitations=[
            "Requires exactly one optical and one SAR image of the same area; refuses otherwise.",
            "SAR is treated as log-scaled, uncalibrated backscatter unless metadata declares "
            "calibration; sigma0/gamma0 is never assumed.",
            "Reports measured backscatter/roughness for SAR, not semantic land-cover classes.",
            "NDVI/NDWI and optical-SAR water agreement need the optical raster to declare bands.",
            "The optional land-cover reading is optical-only, zero-shot, and indicative.",
        ],
    ),
    # Support tools (not directly user-facing tasks)
    # Implemented in Phase 2 (geospatial package + ingestion service).
    "raster_preprocessor": ToolSpec(
        id="raster_preprocessor",
        name="RasterPreprocessor",
        version="1.0.0",
        task=TaskType.METADATA_INSPECTION,
        modalities=[Modality.ANY],
        input_contract="{ image_id: str }",
        output_contract="{ metadata: RasterMetadata, preview_ref: str|null }",
        allowed_parameters=[
            ParamSpec(
                name="max_preview_side",
                type="int",
                default=1024,
                minimum=128,
                maximum=4096,
                description="Longest side of the rendered preview in pixels.",
            )
        ],
        device="cpu",
        checkpoint=None,
        license="OSS (Rasterio/GDAL)",
        status=ToolStatus.AVAILABLE,
        phase=2,
    ),
    "registration_validator": ToolSpec(
        id="registration_validator",
        name="RegistrationValidator",
        version="1.0.0",
        task=TaskType.METADATA_INSPECTION,
        modalities=[Modality.ANY],
        input_contract="{ image_id_a: str, image_id_b: str, purpose: 'temporal'|'optical_sar' }",
        output_contract=(
            "{ compatible: bool, issues: Issue[], overlap_ratio: float|null, "
            "resampling_required: bool, alignment_required: bool }"
        ),
        allowed_parameters=[],
        device="cpu",
        checkpoint=None,
        license="OSS (Rasterio/pyproj)",
        status=ToolStatus.AVAILABLE,
        phase=2,
    ),
    "evidence_aggregator": ToolSpec(
        id="evidence_aggregator",
        name="EvidenceAggregator",
        version="0.0.0",
        task=TaskType.METADATA_INSPECTION,
        modalities=[Modality.ANY],
        input_contract="{ specialist_outputs: Any[] }",
        output_contract="{ evidence: Evidence[] }",
        allowed_parameters=[],
        device="cpu",
        checkpoint=None,
        license="N/A",
        status=ToolStatus.PLANNED,
        phase=4,
    ),
    "confidence_calibrator": ToolSpec(
        id="confidence_calibrator",
        name="ConfidenceCalibrator",
        version="0.0.0",
        task=TaskType.METADATA_INSPECTION,
        modalities=[Modality.ANY],
        input_contract="{ signals: ConfidenceSignals }",
        output_contract="{ score: float, level: 'High'|'Medium'|'Low'|'Unable' }",
        allowed_parameters=[],
        device="cpu",
        checkpoint=None,
        license="N/A",
        status=ToolStatus.PLANNED,
        phase=4,
    ),
}


def list_tools() -> list[ToolSpec]:
    """Return all registered tools."""
    return list(MODEL_REGISTRY.values())


def tools_for_task(task: TaskType) -> list[ToolSpec]:
    """Return tools that serve a given task."""
    return [t for t in MODEL_REGISTRY.values() if t.task == task]


def get_registry_health() -> dict[str, str]:
    """Summarize each tool's current status for health/readiness reporting."""
    return {tool_id: spec.status.value for tool_id, spec in MODEL_REGISTRY.items()}
