"""External benchmark dataset registry and availability checks.

This module does **not** download anything. It records, for each public benchmark the SIH
problem statement references, where it comes from, its licence, where this repo expects it
locally, and whether it is currently present. Benchmark scripts use it to decide between
running a real evaluation and emitting an honest ``NOT EVALUATED — REASON`` report.

Keeping this list explicit means a "blocked" benchmark states precisely *why* (dataset not
provisioned) and *how* to provision it, rather than silently producing a fabricated score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """Provenance and local-availability contract for one external dataset."""

    key: str
    name: str
    task: str
    source: str
    license: str
    #: Path (relative to repo root) where the prepared dataset is expected.
    expected_path: str
    #: A short note on why the dataset is or is not directly consumable by the current system.
    compatibility_note: str
    #: Approximate download size, for planning.
    approx_size: str

    def local_path(self, repo_root: Path) -> Path:
        return repo_root / self.expected_path

    def is_provisioned(self, repo_root: Path) -> bool:
        """True only when the expected local path exists and is non-empty."""
        path = self.local_path(repo_root)
        if not path.exists():
            return False
        if path.is_dir():
            return any(path.iterdir())
        return path.stat().st_size > 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


#: The public datasets referenced by SIH26167 plus the standard change-detection sets.
#: BigEarthNet.txt is the one provisioned in this repo (Phase 5). The rest are declared with
#: honest compatibility notes and are provisioned on demand by a developer running the
#: benchmark; they are intentionally NOT downloaded by tests or by default.
DATASETS: dict[str, DatasetSpec] = {
    "bigearthnet_txt": DatasetSpec(
        key="bigearthnet_txt",
        name="BigEarthNet.txt (Lithuania/Summer captioning subset)",
        task="image-text retrieval (adaptation + evaluation)",
        source="Hugging Face / BigEarthNet v2 imagery + BigEarthNet.txt captions",
        license="Imagery CDLA-Permissive-1.0; captions per BigEarthNet.txt release",
        expected_path="data/bigearthnet/prepared/test.jsonl",
        compatibility_note=(
            "Provisioned. Used by the Phase 5 LoRA adaptation and its held-out retrieval "
            "evaluation. Retrieval metrics only — never VQA/caption accuracy."
        ),
        approx_size="~2.5 GB (LMDB) + captions",
    ),
    "rsvqa": DatasetSpec(
        key="rsvqa",
        name="RSVQA (Low/High Resolution)",
        task="remote-sensing visual question answering",
        source="https://rsvqa.sylvainlobry.com/ (Zenodo)",
        license="See RSVQA release (research use)",
        expected_path="data/benchmarks/rsvqa",
        compatibility_note=(
            "Answer format is yes/no, counts, and area/comparison classes. The current VQA "
            "specialist is zero-shot CLIP concept-scoring over a fixed vocabulary and cannot "
            "emit these answer types, so a direct accuracy would be near-zero and "
            "misrepresentative. A constrained/generative VQA head is required for a valid run."
        ),
        approx_size="RSVQA-LR ~ hundreds of MB; RSVQA-HR larger",
    ),
    "vrsbench": DatasetSpec(
        key="vrsbench",
        name="VRSBench",
        task="captioning + grounding + VQA",
        source="https://vrsbench.github.io/ (Hugging Face)",
        license="See VRSBench release",
        expected_path="data/benchmarks/vrsbench",
        compatibility_note=(
            "Grounding (box IoU) is compatible with the OWL-ViT zero-shot grounding specialist "
            "if annotations are provisioned. Captioning expects generated natural-language "
            "captions; the current captioner is retrieval-composed and cannot produce valid "
            "generative-caption predictions. VQA has the same answer-format limitation as RSVQA."
        ),
        approx_size="~GBs",
    ),
    "cdvqa": DatasetSpec(
        key="cdvqa",
        name="CDVQA (Change Detection VQA)",
        task="bi-temporal change visual question answering",
        source="https://github.com/YZHJessica/CDVQA (dataset release)",
        license="See CDVQA release",
        expected_path="data/benchmarks/cdvqa",
        compatibility_note=(
            "The measurement-grounded change-VQA specialist answers occurrence/location/"
            "extent/NDVI-NDWI-direction questions. It declines semantic-class questions "
            "(e.g. building type) by design. The published CDVQA field schema is unverified "
            "here; a documented JSONL adapter (ml.evaluation.change_vqa.load_samples) converts "
            "a real release. Coverage is always reported next to accuracy."
        ),
        approx_size="~GBs",
    ),
    "second": DatasetSpec(
        key="second",
        name="SECOND (SEmantic Change detectiON Dataset)",
        task="semantic / from-to change detection",
        source=(
            "Additional public dataset (NOT SIH-provided). User-provided source: "
            "https://drive.usercontent.google.com/download?id=1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc"
        ),
        license="See SECOND dataset release (research use); verify before redistribution",
        expected_path="data/benchmarks/second",
        compatibility_note=(
            "ADDITIONAL PUBLIC dataset for external validation/training of change detection — "
            "explicitly NOT an SIH26167-prescribed dataset. Provides semantic change masks "
            "(from-to land-cover). The current deterministic detector produces BINARY "
            "radiometric change only; using SECOND for binary change requires collapsing "
            "semantic labels to changed/unchanged (a documented, lossy conversion). A learned "
            "semantic detector would be a new component; the Phase 6 detector is not removed."
        ),
        approx_size="~GB (SECOND_train_set.rar)",
    ),
    "levir_cd": DatasetSpec(
        key="levir_cd",
        name="LEVIR-CD",
        task="binary change detection",
        source="https://chenhao.in/LEVIR/",
        license="See LEVIR-CD release (research use)",
        expected_path="data/benchmarks/levir_cd",
        compatibility_note=(
            "Building-change dataset with binary masks. The change detector (cva-robust) "
            "measures radiometric change, not building semantics, so it is expected to flag "
            "broad radiometric differences rather than only building changes; scores must be "
            "read with that caveat. Pairs are co-registered RGB, consumable via align_pair."
        ),
        approx_size="~1-2 GB",
    ),
    "oscd": DatasetSpec(
        key="oscd",
        name="OSCD (Onera Satellite Change Detection)",
        task="binary change detection",
        source="https://rcdaudt.github.io/oscd/",
        license="See OSCD release (research use)",
        expected_path="data/benchmarks/oscd",
        compatibility_note=(
            "Sentinel-2 multi-date pairs with binary change masks. Multi-band GeoTIFFs are "
            "consumable via align_pair; radiometric-vs-semantic caveat applies as for LEVIR-CD."
        ),
        approx_size="~1 GB",
    ),
}


def availability(repo_root: Path) -> dict[str, dict[str, object]]:
    """Report provisioning status for every registered dataset."""
    report: dict[str, dict[str, object]] = {}
    for key, spec in DATASETS.items():
        report[key] = {
            **spec.as_dict(),
            "provisioned": spec.is_provisioned(repo_root),
            "resolved_path": str(spec.local_path(repo_root)),
        }
    return report
