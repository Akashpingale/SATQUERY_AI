"""Phase 12 Flow D - learned temporal route validation.

The learned specialists (`second_change_learned`, `cdvqa_trained`) are registered and
invocable but are deliberately NOT on any default task toolchain: the agent never
auto-selects them (the deterministic Phase 6 CVA stays the production default). This
script validates the learned route *on the imagery it was actually trained on* - the
public SECOND aerial RGB PNGs - proving the checkpoints load, inference runs, and the
outputs are valid, without integrating them into the default workflow.

It does NOT retrain, does NOT change any default, and does NOT touch checkpoints.

Run:
    $env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
    $env:HF_HOME="D:\\SatQuery-AI\\models\\hf-cache"; $env:DEVICE="cpu"
    .\\.venv\\Scripts\\python.exe scripts\\phase12\\flow_d_learned.py --pairs 3

Output: reports/phase12/e2e/flow_d_learned.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PHASE_E2E_OUT", str(REPO / "reports" / "phase12" / "e2e")))
if not OUT.is_absolute():
    OUT = REPO / OUT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--second-root", default=r"D:\dataset\second")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument(
        "--question", default="Did new buildings appear?", help="CDVQA question to ask."
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO / "backend"))
    sys.path.insert(0, str(REPO))

    # Imports are deferred until after sys.path is set so `app` and `ml` resolve.
    from app.agent.planner import TASK_TOOLCHAIN  # noqa: I001
    from ml.inference import cdvqa_infer, second_infer

    # 1. Routing guard: learned tools must NOT be on any default task toolchain.
    default_tools = {tool for chain in TASK_TOOLCHAIN.values() for tool in chain}
    learned_default = sorted({"second_change_learned", "cdvqa_trained"} & default_tools)

    report: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "Validate the non-default learned temporal route (SECOND + CDVQA).",
        "routing_guard": {
            "task_toolchain": {str(k): v for k, v in TASK_TOOLCHAIN.items()},
            "learned_tools_on_default_toolchain": learned_default,
            "deterministic_remains_default": learned_default == [],
        },
        "availability": {
            "second_change_learned": second_infer.availability().__dict__,
            "cdvqa_trained": cdvqa_infer.availability().__dict__,
        },
    }

    im1_dir = Path(args.second_root) / "im1"
    im2_dir = Path(args.second_root) / "im2"
    names = sorted(p.name for p in im1_dir.glob("*.png") if (im2_dir / p.name).is_file())
    sample = names[: args.pairs]

    # 2. SECOND learned binary change detector - checkpoint loads + valid output in [0,1].
    second_runs = []
    for name in sample:
        pred = second_infer.predict(str(im1_dir / name), str(im2_dir / name))
        second_runs.append(
            {
                "pair": name,
                "changed_fraction": pred.changed_fraction,
                "mean_confidence": pred.mean_confidence,
                "analysis_px": pred.size,
                "model_version": pred.model_version,
                "output_valid": 0.0 <= pred.changed_fraction <= 1.0,
            }
        )
    report["second_change_learned_runs"] = second_runs
    report["second_checkpoint_loaded"] = bool(second_runs)

    # 3. CDVQA trained classifier - closed-vocabulary answer + softmax confidence.
    cdvqa_runs = []
    for name in sample:
        pred = cdvqa_infer.predict(
            str(im1_dir / name), str(im2_dir / name), args.question
        )
        cdvqa_runs.append(
            {
                "pair": name,
                "question": args.question,
                "answer": pred.answer,
                "confidence": round(pred.confidence, 4),
                "top5": [(a, round(p, 4)) for a, p in pred.ranked],
                "model_version": pred.model_version,
            }
        )
    report["cdvqa_trained_runs"] = cdvqa_runs
    report["cdvqa_checkpoint_loaded"] = bool(cdvqa_runs)

    # 4. "Not silently forced onto incompatible imagery": the learned specialists read
    #    RGB via PIL and are scoped to the SECOND-style aerial RGB they were trained on.
    #    The multi-band uint16 GeoTIFF demo fixtures are handled by the deterministic
    #    route instead (see Flow B), which is exactly why deterministic stays default.
    report["notes"] = [
        "Learned route is non-default; the agent's TASK_TOOLCHAIN routes temporal/"
        "change-VQA to the deterministic CVA specialists.",
        "Learned SECOND/CDVQA read RGB via PIL; they target SECOND-style aerial RGB, "
        "not arbitrary multi-band GeoTIFF. Demo GeoTIFFs are served by the deterministic "
        "route (Flow B), confirming the robust default.",
        "Checkpoints were loaded read-only; nothing was retrained or overwritten.",
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flow_d_learned.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
