import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_FAILURE_GALLERY_PATH = "Data/reports/failure_case_gallery.json"


def load_failure_case_gallery(path: str = DEFAULT_FAILURE_GALLERY_PATH) -> dict:
    gallery_path = Path(path)
    if not gallery_path.exists():
        return {
            "schema_version": "failure_case_gallery_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "not_generated",
            "message": "Failure case gallery not generated yet.",
            "cases": [],
        }
    try:
        payload = json.loads(gallery_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "schema_version": "failure_case_gallery_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "message": "Failure case gallery could not be parsed.",
            "cases": [],
        }
    payload.setdefault("status", "available")
    return payload
