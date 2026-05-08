import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.database import SessionLocal  # noqa: E402
from backend.schema_migrations import ensure_schema  # noqa: E402
from backend.services.patient_data_quality import DEFAULT_COHERENCE_AUDIT_PATH, audit_patient_data_coherence  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Audit patient timeline coherence across CBC, treatment cycles, MRI reports, and outcomes.")
    parser.add_argument("--output-path", default=DEFAULT_COHERENCE_AUDIT_PATH)
    args = parser.parse_args()

    ensure_schema()
    db = SessionLocal()
    try:
        report = audit_patient_data_coherence(db, output_path=args.output_path)
    finally:
        db.close()

    print(json.dumps({
        "output_path": args.output_path,
        "status": report["status"],
        "patient_count": report["patient_count"],
        "patients_with_warnings_or_errors": report["patients_with_warnings_or_errors"],
        "issue_counts": report["issue_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
