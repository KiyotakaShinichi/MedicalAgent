import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.database import SessionLocal  # noqa: E402
from backend.models import (  # noqa: E402
    AgentResponseCache,
    AgentResponseFeedback,
    AppEventLog,
    ClinicalSummaryReview,
    MLExperimentRun,
    PredictionAuditLog,
    RAGEvaluationLog,
)
from backend.schema_migrations import ensure_schema  # noqa: E402

DEFAULT_OUTPUT_PATH = ROOT_DIR / "Data" / "agent_eval" / "audit_log_check.json"

TABLES = [
    ("app_event_logs", AppEventLog),
    ("prediction_audit_logs", PredictionAuditLog),
    ("rag_evaluation_logs", RAGEvaluationLog),
    ("clinical_summary_reviews", ClinicalSummaryReview),
    ("agent_response_feedback", AgentResponseFeedback),
    ("agent_response_cache", AgentResponseCache),
    ("ml_experiment_runs", MLExperimentRun),
]


def main():
    parser = argparse.ArgumentParser(description="Verify audit logging tables have data.")
    parser.add_argument("--min-count", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    ensure_schema()
    db = SessionLocal()
    try:
        rows = []
        for name, model in TABLES:
            count = db.query(model).count()
            rows.append({"table": name, "count": count})
    finally:
        db.close()

    status = "passed"
    if args.strict and any(item["count"] <= args.min_count for item in rows):
        status = "failed"

    output = {
        "schema_version": "audit_log_check_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "min_count": args.min_count,
        "tables": rows,
        "claim_boundary": "Audit log checks verify telemetry presence, not clinical correctness.",
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_path": str(output_path.relative_to(ROOT_DIR)),
        "status": status,
        "tables_below_min": [item["table"] for item in rows if item["count"] <= args.min_count],
    }, indent=2))

    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
