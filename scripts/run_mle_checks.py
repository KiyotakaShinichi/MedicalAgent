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
    args = parser.parse_args()

    ensure_schema()
    db = SessionLocal()
    try:
        report = build_mle_readiness_summary(db=db, output_path=args.output_path)
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
