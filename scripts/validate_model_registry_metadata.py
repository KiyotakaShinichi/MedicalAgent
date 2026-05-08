import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.database import SessionLocal  # noqa: E402
from backend.models import ModelRegistry  # noqa: E402
from backend.schema_migrations import ensure_schema  # noqa: E402

DEFAULT_OUTPUT_PATH = ROOT_DIR / "Data" / "model_registry" / "metadata_validation.json"

REQUIRED_METADATA_KEYS = {
    "model_name",
    "model_version",
    "task",
    "promotion_status",
    "registry_schema_version",
    "warning",
}
ALLOWED_STATUSES = {"active", "champion", "archived", "rolled_back"}


def _parse_metadata(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Validate model registry metadata JSON fields.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    ensure_schema()
    db = SessionLocal()
    try:
        rows = db.query(ModelRegistry).order_by(ModelRegistry.created_at.desc()).all()
    finally:
        db.close()

    results = []
    for row in rows:
        metadata = _parse_metadata(row.model_metadata_json)
        missing = sorted(key for key in REQUIRED_METADATA_KEYS if not metadata.get(key))
        mismatches = []
        if metadata.get("model_name") and metadata.get("model_name") != row.model_name:
            mismatches.append("model_name")
        if metadata.get("model_version") and metadata.get("model_version") != row.model_version:
            mismatches.append("model_version")
        status_ok = row.status in ALLOWED_STATUSES
        results.append({
            "id": row.id,
            "model_name": row.model_name,
            "model_version": row.model_version,
            "status": row.status,
            "status_ok": status_ok,
            "missing_metadata_keys": missing,
            "mismatched_fields": mismatches,
            "artifact_path": row.artifact_path,
        })

    failed = [item for item in results if item["missing_metadata_keys"] or item["mismatched_fields"] or not item["status_ok"]]
    status = "passed" if not failed else "failed"

    output = {
        "schema_version": "model_registry_metadata_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "row_count": len(results),
        "failed_rows": len(failed),
        "rows": results,
        "claim_boundary": "Registry metadata validation checks consistency, not clinical performance.",
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_path": str(output_path.relative_to(ROOT_DIR)),
        "status": status,
        "failed_rows": len(failed),
    }, indent=2))

    if args.strict and status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
