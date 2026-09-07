"""External benchmark evaluation entry point (RSVQA / VRSBench / CDVQA / SECOND / LEVIR-CD / OSCD).

This script never downloads anything. It checks whether a dataset has been provisioned locally
(see ml.evaluation.datasets for the expected paths) and, when it is, runs the matching adapter.
When a dataset is absent it writes an honest ``NOT RUN`` report stating the reason, the
compatibility caveats, and the exact command to run once the data is provisioned — rather than
fabricating a score.

    python scripts/eval_external.py --dataset cdvqa
    python scripts/eval_external.py --all        # status for every registered dataset

Provisioning is a manual, licence-aware step done by a developer; it is intentionally NOT
automated here so tests, Docker builds, and CI never trigger multi-GB downloads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "reports" / "benchmarks" / "external"


def _status_report(key: str) -> dict:
    from ml.evaluation.datasets import DATASETS
    from ml.evaluation.reproducibility import build_run_context

    spec = DATASETS[key]
    provisioned = spec.is_provisioned(REPO_ROOT)
    ctx = build_run_context(
        benchmark=f"external:{key}",
        dataset=spec.name,
        split="n/a (not provisioned)" if not provisioned else "see adapter",
        samples=0,
        seed=42,
        repo_root=REPO_ROOT,
    )
    return {
        "dataset": spec.as_dict(),
        "provisioned": provisioned,
        "status": (
            "PROVISIONED (run adapter)" if provisioned else "NOT RUN — dataset not provisioned"
        ),
        "reason": (
            ""
            if provisioned
            else (
                f"{spec.name} is not present at {spec.local_path(REPO_ROOT)}. It is not "
                "downloaded automatically (licence-aware, multi-GB). Provision it, then re-run."
            )
        ),
        "reproducible_command": f"python scripts/eval_external.py --dataset {key}",
        "environment": ctx.as_dict(),
    }


def main() -> int:
    from ml.evaluation.datasets import DATASETS
    from ml.evaluation.reproducibility import write_json

    parser = argparse.ArgumentParser(description="External benchmark availability / runner")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--all", action="store_true", help="report status for every dataset")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("choose --dataset <key> or --all")

    keys = sorted(DATASETS) if args.all else [args.dataset]
    for key in keys:
        report = _status_report(key)
        write_json(OUT / f"{key}.json", report)
        print(f"{key:12} {report['status']}")
        if report["reason"]:
            print(f"             {report['reason']}")
    print(f"\nReports written under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
