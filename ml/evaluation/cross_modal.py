"""Cross-modal (optical+SAR) engineering-validation harness.

Runs the **production** feature-level fusion (`ml.inference.analyze_cross_modal` via
`geospatial.align_modalities`) on deterministic synthetic fixtures and checks that the measured
quantities respond correctly to controlled inputs.

This is engineering validation of the shipped algorithm — NOT a real-world benchmark. No public
optical+SAR cross-modal "complementarity" benchmark matching this exact task is used (see
docs/CROSS_MODAL_EVALUATION_ABSENCE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geospatial import align_modalities, read_metadata, resolve_band_roles
from ml.evaluation.synthetic import build_cross_modal_fixtures
from ml.inference.fusion import analyze_cross_modal


@dataclass
class CrossModalCheck:
    key: str
    expectation: str
    passed: bool
    detail: str
    stats: dict[str, Any]


def _analyze(optical_path: str, sar_path: str) -> Any:
    aligned = align_modalities(optical_path, sar_path)
    optical_roles = resolve_band_roles(read_metadata(optical_path))
    return analyze_cross_modal(aligned, optical_roles=optical_roles, sar_tags={})


def run(out_dir: str | Path) -> dict[str, Any]:
    """Build fixtures, run production fusion, and verify measurable properties.

    Returns a structured result with per-scenario stats and pass/fail checks. Passing means
    the algorithm's outputs move in the expected direction under controlled inputs; it does
    not assert absolute accuracy against any real dataset.
    """
    out = Path(out_dir)
    scenarios = build_cross_modal_fixtures(out / "fixtures")
    results = {s.key: _analyze(s.optical_path, s.sar_path) for s in scenarios}
    stats = {key: res.stats_dict() for key, res in results.items()}

    checks: list[CrossModalCheck] = []

    def add(key: str, expectation: str, passed: bool, detail: str) -> None:
        checks.append(CrossModalCheck(key, expectation, passed, detail, stats.get(key, {})))

    corr = results["correlated"]
    anti = results["anticorrelated"]
    # Complementarity = 1 - |correlation|. Correlated pair should be less complementary.
    if corr.complementarity is not None and anti.complementarity is not None:
        passed = corr.complementarity < anti.complementarity
        add(
            "correlated",
            "correlated pair less complementary than anticorrelated",
            passed,
            f"corr.complementarity={corr.complementarity} < anti={anti.complementarity}",
        )
    else:
        add(
            "correlated",
            "complementarity measured for both correlated/anticorrelated pairs",
            False,
            f"correlation withheld (corr={corr.correlation}, anti={anti.correlation})",
        )

    add(
        "anticorrelated",
        "anticorrelated pair is strongly complementary (>= 0.5)",
        anti.complementarity is not None and anti.complementarity >= 0.5,
        f"complementarity={anti.complementarity}",
    )

    sar = results["sar_structure"]
    flat_sar = results["correlated"].sar.structured_fraction
    add(
        "sar_structure",
        "SAR-only information/structure exceeds the flat-SAR baseline",
        sar.sar_only_information > 0.0 or sar.sar.structured_fraction > flat_sar,
        f"sar_only_information={sar.sar_only_information}, "
        f"structured_fraction={sar.sar.structured_fraction} vs flat={flat_sar}",
    )

    water = results["water_agreement"].water
    add(
        "water_agreement",
        "cross-modal water agreement available with non-zero overlap",
        bool(water.available) and water.agree_fraction > 0.0,
        f"available={water.available}, agree_fraction={water.agree_fraction}, "
        f"iou={water.agreement_iou}",
    )

    passed = sum(1 for c in checks if c.passed)
    return {
        "benchmark": "cross_modal_synthetic",
        "kind": "engineering-validation (synthetic; not a real-world benchmark)",
        "algorithm_version": next(iter(results.values())).algorithm_version,
        "scenarios": len(scenarios),
        "checks_total": len(checks),
        "checks_passed": passed,
        "all_passed": passed == len(checks),
        "checks": [c.__dict__ for c in checks],
        "stats": stats,
    }
