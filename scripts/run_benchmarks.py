"""Phase 9 offline benchmark + calibration runner.

Executes the evaluations that are genuinely runnable in this environment without any external
download, routing through the REAL production specialists:

  - cross_modal        : synthetic optical+SAR fusion property validation
  - change_detection   : synthetic change detection vs known masks (IoU/P/R/F1)
  - change_vqa         : synthetic change-occurrence VQA accuracy/coverage
  - calibration        : confidence reliability/ECE/Brier/selective-accuracy measured on the
                         REAL end-to-end pipeline (agent orchestrator) over the change-VQA set

These are engineering-validation results on synthetic fixtures — NOT real-world benchmark
accuracy. Real benchmarks (RSVQA, VRSBench, CDVQA, SECOND, LEVIR-CD, OSCD) require their
datasets; see scripts/eval_external.py and docs/benchmarking.md.

    python scripts/run_benchmarks.py --suite all
    python scripts/run_benchmarks.py --suite calibration --seed 42

Outputs are written under reports/benchmarks/<suite>/.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

# Offline + isolated storage/DB BEFORE any app import (settings are cached at import time).
_TMP = Path(tempfile.mkdtemp(prefix="satquery_bench_"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(_TMP / 'bench.db').as_posix()}")
os.environ.setdefault("STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DEVICE", "cpu")

OUT_ROOT = REPO_ROOT / "reports" / "benchmarks"


def _reproducibility():
    from ml.evaluation.reproducibility import build_run_context, write_json, write_jsonl

    return build_run_context, write_json, write_jsonl


def run_cross_modal() -> dict:
    from ml.evaluation import cross_modal
    from ml.evaluation.reproducibility import build_run_context, write_json

    out = OUT_ROOT / "cross_modal"
    result = cross_modal.run(out)
    ctx = build_run_context(
        benchmark="cross_modal_synthetic",
        dataset="synthetic optical+SAR fixtures",
        split="n/a (synthetic)",
        samples=result["scenarios"],
        seed=0,
        repo_root=REPO_ROOT,
        models={"fusion": result["algorithm_version"]},
    )
    write_json(out / "metrics.json", result)
    write_json(out / "config.json", ctx.as_dict())
    _write_md(
        out / "report.md",
        "Cross-modal (optical+SAR) synthetic validation",
        result["kind"],
        [
            f"- Scenarios: {result['scenarios']}",
            f"- Property checks passed: {result['checks_passed']}/{result['checks_total']}",
            f"- Fusion algorithm: {result['algorithm_version']}",
            "",
            "Per-check:",
            *[
                f"  - [{'PASS' if c['passed'] else 'FAIL'}] {c['key']}: {c['expectation']} "
                f"({c['detail']})"
                for c in result["checks"]
            ],
        ],
    )
    return result


def run_change_detection() -> dict:
    from ml.evaluation import change_detection
    from ml.evaluation.reproducibility import build_run_context, write_json, write_jsonl

    out = OUT_ROOT / "change_detection"
    result = change_detection.run(out)
    ctx = build_run_context(
        benchmark="change_detection_synthetic",
        dataset="synthetic bi-temporal fixtures (known masks)",
        split="n/a (synthetic)",
        samples=result["scenarios"],
        seed=0,
        repo_root=REPO_ROOT,
        models={"detector": result["detector"]},
    )
    write_json(out / "metrics.json", result)
    write_json(out / "config.json", ctx.as_dict())
    write_jsonl(out / "predictions.jsonl", result["predictions"])
    m = result["metrics"]
    _write_md(
        out / "report.md",
        "Change detection synthetic validation",
        result["kind"],
        [
            f"- Detector: {result['detector']}",
            f"- Scenarios: {result['scenarios']}",
            f"- micro IoU: {m['micro_iou']}  precision: {m['precision']}  recall: {m['recall']}"
            f"  F1: {m['f1']}",
            f"- mean IoU (changed pairs): {m['mean_iou_changed']}",
            f"- occurrence accuracy: {m['occurrence_accuracy']}",
            f"- pixels: {m['pixels']}",
            "",
            result["note"],
        ],
    )
    return result


def _occurrence_samples():
    from ml.evaluation.change_vqa import ChangeVQASample
    from ml.evaluation.synthetic import build_change_vqa_fixtures

    raw = build_change_vqa_fixtures(OUT_ROOT / "change_vqa" / "fixtures")
    samples = [
        ChangeVQASample(
            sample_id=s.sample_id,
            before_path=s.t1_path,
            after_path=s.t2_path,
            question="Did anything change between these two images?",
            answer="yes" if s.changed else "no",
            question_type=f"occurrence:{s.magnitude}",
        )
        for s in raw
    ]
    return raw, samples


def run_change_vqa() -> dict:
    from ml.evaluation.change_vqa import predict_all, score_predictions
    from ml.evaluation.metrics import accuracy_ci
    from ml.evaluation.reproducibility import build_run_context, write_json, write_jsonl

    out = OUT_ROOT / "change_vqa"
    _raw, samples = _occurrence_samples()
    predictions = predict_all(samples)
    metrics = score_predictions(
        samples,
        predictions,
        note="Synthetic change-occurrence VQA (engineering validation; not CDVQA).",
    )
    ci = accuracy_ci(metrics.correct, metrics.samples)

    ctx = build_run_context(
        benchmark="change_vqa_synthetic",
        dataset="synthetic change-occurrence fixtures",
        split="n/a (synthetic)",
        samples=metrics.samples,
        seed=0,
        repo_root=REPO_ROOT,
        models={"change_vqa": "cva-robust (deterministic)"},
    )
    payload = {
        "benchmark": "change_vqa_synthetic",
        "kind": "engineering-validation (synthetic occurrence; not CDVQA)",
        "metrics": metrics.as_dict(),
        "accuracy_ci_wilson": ci,
    }
    write_json(out / "metrics.json", payload)
    write_json(out / "config.json", ctx.as_dict())
    write_jsonl(out / "predictions.jsonl", [p.as_dict() for p in predictions])
    _write_md(
        out / "report.md",
        "Change-occurrence VQA synthetic validation",
        payload["kind"],
        [
            f"- Samples: {metrics.samples}  answered: {metrics.answered}  "
            f"correct: {metrics.correct}",
            f"- Accuracy: {metrics.accuracy}  (Wilson 95% CI {ci['ci_low']}–{ci['ci_high']})",
            f"- Accuracy on answered: {metrics.accuracy_on_answered}  coverage: {metrics.coverage}",
            "",
            "Per question-type:",
            *[
                f"  - {k}: acc={v['accuracy']} coverage={v['coverage']} (n={int(v['samples'])})"
                for k, v in metrics.per_type.items()
            ],
        ],
    )
    return payload


def run_calibration() -> dict:
    """Measure confidence calibration on the REAL end-to-end pipeline (agent orchestrator).

    Each synthetic occurrence pair is ingested into an isolated temp DB and analysed with
    ``run_analysis`` (the same path the API uses). The production confidence score is paired
    with an independent yes/no correctness label to compute reliability/ECE/Brier/selective
    accuracy. Synthetic fixture, small N — reported honestly; thresholds are NOT re-tuned here.
    """
    import app.models_db  # noqa: F401  (register ORM models)
    from app.agent import run_analysis
    from app.db.base import Base
    from app.db.session import SessionLocal, engine
    from app.services.ingestion import ingest_upload

    from ml.evaluation.calibration import abstention_quality, reliability, selective_accuracy
    from ml.evaluation.change_vqa import predict_sample
    from ml.evaluation.reproducibility import build_run_context, write_json, write_jsonl

    Base.metadata.create_all(bind=engine)
    out = OUT_ROOT / "calibration"
    raw, samples = _occurrence_samples()
    by_id = {s.sample_id: s for s in samples}

    def ingest(db, path: str, role: str) -> str:
        with open(path, "rb") as handle:
            data = handle.read()
        result = ingest_upload(
            db,
            io.BytesIO(data),
            Path(path).name,
            content_type="image/tiff",
            role=role,
            modality="optical",
        )
        if not result.ok or result.asset is None:
            raise RuntimeError(f"ingestion failed for {path}")
        return result.asset.id

    records: list[dict] = []
    confidences: list[float] = []
    correct: list[bool] = []
    answered_flags: list[bool] = []
    correct_if_answered: list[bool] = []

    db = SessionLocal()
    try:
        for occ in raw:
            sample = by_id[occ.sample_id]
            t1_id = ingest(db, occ.t1_path, "t1")
            t2_id = ingest(db, occ.t2_path, "t2")
            outcome = run_analysis(db, sample.question, [t1_id, t2_id])
            short = predict_sample(sample)  # independent yes/no via the same measurement
            is_correct = short.answered and short.predicted == sample.answer
            conf = outcome.confidence.score
            answered = conf is not None and outcome.status.value == "completed"

            answered_flags.append(bool(answered))
            correct_if_answered.append(bool(is_correct))
            if answered:
                confidences.append(float(conf))
                correct.append(bool(is_correct))

            records.append(
                {
                    "sample_id": occ.sample_id,
                    "truth": sample.answer,
                    "predicted": short.predicted,
                    "answered": bool(answered),
                    "correct": bool(is_correct),
                    "confidence": conf,
                    "status": outcome.status.value,
                    "magnitude": occ.magnitude,
                }
            )
    finally:
        db.close()

    calib = reliability(
        confidences,
        correct,
        n_bins=5,
        note="Synthetic change-occurrence set; production end-to-end confidence. Small N.",
    )
    selective = selective_accuracy(confidences, correct)
    abstention = abstention_quality(answered_flags, correct_if_answered)

    ctx = build_run_context(
        benchmark="confidence_calibration_synthetic",
        dataset="synthetic change-occurrence (end-to-end via orchestrator)",
        split="n/a (synthetic); thresholds not re-tuned",
        samples=len(records),
        seed=0,
        repo_root=REPO_ROOT,
        models={"confidence": "app.agent.confidence.assess (HIGH>=0.75, MEDIUM>=0.45)"},
    )
    payload = {
        "benchmark": "confidence_calibration_synthetic",
        "kind": "engineering-validation calibration (synthetic; small N; thresholds unchanged)",
        "answered": len(confidences),
        "calibration": calib.as_dict(),
        "selective_accuracy": [p.__dict__ for p in selective],
        "abstention": abstention.as_dict(),
    }
    write_json(out / "metrics.json", payload)
    write_json(out / "config.json", ctx.as_dict())
    write_jsonl(out / "predictions.jsonl", records)
    _write_md(
        out / "report.md",
        "Confidence calibration (synthetic, end-to-end)",
        payload["kind"],
        [
            f"- Samples: {len(records)}  answered: {len(confidences)}",
            f"- ECE: {calib.ece}  MCE: {calib.mce}  Brier: {calib.brier}",
            f"- Overall confidence: {calib.overall_confidence}  overall accuracy: "
            f"{calib.overall_accuracy}",
            f"- Coverage: {abstention.coverage}  selective accuracy: "
            f"{abstention.selective_accuracy}",
            "",
            "Reliability bins (lower–upper: n, mean_conf, acc):",
            *[
                f"  - {b['lower']}–{b['upper']}: n={b['count']} conf={b['mean_confidence']} "
                f"acc={b['accuracy']}"
                for b in calib.as_dict()["bins"]
            ],
            "",
            "Interpretation: with a small synthetic N this measures whether the production "
            "confidence tracks correctness on this fixture; it is not a benchmark calibration "
            "and no thresholds were changed on the basis of it.",
        ],
    )
    return payload


def _write_md(path: Path, title: str, kind: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines)
    path.write_text(f"# {title}\n\n**Kind:** {kind}\n\n{body}\n", encoding="utf-8")


SUITES = {
    "cross_modal": run_cross_modal,
    "change_detection": run_change_detection,
    "change_vqa": run_change_vqa,
    "calibration": run_calibration,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 offline benchmark runner")
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    suites = list(SUITES) if args.suite == "all" else [args.suite]
    summary: dict[str, str] = {}
    for name in suites:
        print(f"=== running {name} ===", flush=True)
        try:
            result = SUITES[name]()
            summary[name] = "ok"
            print(f"  {name}: done", flush=True)
            _ = result
        except Exception as exc:  # noqa: BLE001 - report which suite failed, keep going
            summary[name] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"  {name}: {summary[name]}", flush=True)

    print("\n=== summary ===")
    for name, status in summary.items():
        print(f"  {name}: {status}")
    return 0 if all(v == "ok" for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
