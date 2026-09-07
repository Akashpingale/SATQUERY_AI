"""Specialist implementations and the lookup used by the executor.

A registry entry existing does NOT mean a specialist is runnable. Only tool ids present
in :data:`SPECIALISTS` have implementations; anything else raises
:class:`SpecialistUnavailable` so the system reports the truth instead of inventing an
answer. As later phases land, their specialists are added here and their registry status
flips accordingly.
"""

from __future__ import annotations

from .base import Specialist, SpecialistContext, SpecialistUnavailable
from .captioning import CaptioningSpecialist
from .captioning_blip import CaptioningBlipSpecialist
from .cdvqa_trained import CDVQATrainedSpecialist
from .cdvqa_transfer import CDVQATransferSpecialist
from .change_vqa_blip import ChangeVQABlipSpecialist
from .fusion import OpticalSARFusionSpecialist
from .grounding import GroundingSpecialist
from .metadata_inspection import (
    RasterPreprocessorSpecialist,
    RegistrationValidatorSpecialist,
)
from .second_change import SecondChangeSpecialist
from .temporal import ChangeVQASpecialist, TemporalChangeSpecialist
from .vqa import VQASpecialist

SPECIALISTS: dict[str, Specialist] = {
    RasterPreprocessorSpecialist.tool_id: RasterPreprocessorSpecialist(),
    RegistrationValidatorSpecialist.tool_id: RegistrationValidatorSpecialist(),
    # Phase 4 baselines and the Phase 5 adapted variants are separate registry entries, so
    # a result always names which model produced it.
    "vqa": VQASpecialist("vqa", variant="baseline"),
    "vqa_adapted": VQASpecialist("vqa_adapted", variant="adapted"),
    "captioning": CaptioningSpecialist("captioning", variant="baseline"),
    "captioning_adapted": CaptioningSpecialist("captioning_adapted", variant="adapted"),
    # External BLIP generative captioner (caption_model/), fine-tuned on VRSBench. Added as an
    # ADDITIONAL captioning specialist; the CLIP-based captioning stays available as a fallback.
    CaptioningBlipSpecialist.tool_id: CaptioningBlipSpecialist(),
    GroundingSpecialist.tool_id: GroundingSpecialist(),
    # Phase 6. Both read one algorithmic change measurement; neither needs model weights,
    # so temporal change works on any machine including CI.
    TemporalChangeSpecialist.tool_id: TemporalChangeSpecialist(),
    ChangeVQASpecialist.tool_id: ChangeVQASpecialist(),
    # Phase 7. Algorithmic optical+SAR cross-modal analysis; no weights required.
    OpticalSARFusionSpecialist.tool_id: OpticalSARFusionSpecialist(),
    # Phase 9. Trained CDVQA classifier (real checkpoint). Registered and invocable, but not
    # the default change-VQA route (see cdvqa_trained.py / phase9_model_selection.md).
    CDVQATrainedSpecialist.tool_id: CDVQATrainedSpecialist(),
    # Phase 9. Learned SECOND binary change detector (real checkpoint). Registered/invocable,
    # benchmarked; not the default change route (deterministic CVA stays default).
    SecondChangeSpecialist.tool_id: SecondChangeSpecialist(),
    # Phase 14. External BLIP Change-VQA checkpoint (change_model/), integrated as an
    # ADDITIONAL generative Change-VQA specialist. Registered/invocable; NOT the default
    # (deterministic change_vqa stays default). Reachable when CHANGE_VQA_MODEL selects it.
    ChangeVQABlipSpecialist.tool_id: ChangeVQABlipSpecialist(),
    # Phase 15. Improved CDVQA classifier (CLIP-rsicd + change_model BLIP features), the best
    # learned Change-VQA model (frozen Test 0.7156 > Phase 9 0.7114). Registered/invocable;
    # still non-default (deterministic stays default). Reachable via CHANGE_VQA_MODEL.
    CDVQATransferSpecialist.tool_id: CDVQATransferSpecialist(),
}


def get_specialist(tool_id: str) -> Specialist:
    """Return the implementation for a tool id.

    Raises:
        SpecialistUnavailable: when no implementation exists yet.
    """
    specialist = SPECIALISTS.get(tool_id)
    if specialist is None:
        raise SpecialistUnavailable(tool_id)
    return specialist


def is_implemented(tool_id: str) -> bool:
    return tool_id in SPECIALISTS


__all__ = [
    "SPECIALISTS",
    "CaptioningBlipSpecialist",
    "CDVQATrainedSpecialist",
    "CDVQATransferSpecialist",
    "ChangeVQABlipSpecialist",
    "SecondChangeSpecialist",
    "CaptioningSpecialist",
    "ChangeVQASpecialist",
    "GroundingSpecialist",
    "OpticalSARFusionSpecialist",
    "RasterPreprocessorSpecialist",
    "RegistrationValidatorSpecialist",
    "Specialist",
    "SpecialistContext",
    "SpecialistUnavailable",
    "TemporalChangeSpecialist",
    "VQASpecialist",
    "get_specialist",
    "is_implemented",
]
