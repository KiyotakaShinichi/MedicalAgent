import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.schema_migrations import ensure_schema  # noqa: E402
from backend.database import SessionLocal  # noqa: E402
from backend.services.mle_readiness import DEFAULT_OUTPUT_PATH, build_mle_readiness_summary  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run MLE readiness gates for data, artifacts, model quality, and lifecycle.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--training-csv", default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--evaluation-manifest-path", default=None)
    parser.add_argument("--lineage-path", default=None)
    parser.add_argument("--leakage-audit-path", default=None)
    parser.add_argument("--locked-holdout-path", default=None)
    args = parser.parse_args()

    ensure_schema()
    db = SessionLocal()
    try:
        kwargs = {"db": db, "output_path": args.output_path}
        for arg_name, kwarg_name in [
            ("training_csv", "training_csv"),
            ("metrics_path", "metrics_path"),
            ("predictions_path", "predictions_path"),
            ("evaluation_manifest_path", "evaluation_manifest_path"),
            ("lineage_path", "lineage_path"),
            ("leakage_audit_path", "leakage_audit_path"),
            ("locked_holdout_path", "locked_holdout_path"),
        ]:
            value = getattr(args, arg_name)
            if value:
                kwargs[kwarg_name] = value
        report = build_mle_readiness_summary(**kwargs)
    finally:
        db.close()

    print(json.dumps({
        "output_path": args.output_path,
        "status": report["status"],
        "release_recommendation": report["release_recommendation"],
        "hard_gate_status": report["hard_gate_status"],
        "hard_gate_failures": len(report["hard_gate_failures"]),
        "poc_demo_readiness": report.get("poc_demo_readiness", {}).get("status"),
        "poc_demo_recommendation": report.get("poc_demo_readiness", {}).get("recommendation"),
        "category_statuses": report["category_statuses"],
    }, indent=2))

    if report["hard_gate_status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
