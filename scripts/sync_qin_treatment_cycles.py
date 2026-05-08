import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.database import SessionLocal  # noqa: E402
from backend.schema_migrations import ensure_schema  # noqa: E402
from backend.services.qin_treatment_sync import sync_qin_treatment_cycles  # noqa: E402


def main():
    ensure_schema()
    db = SessionLocal()
    try:
        result = sync_qin_treatment_cycles(db)
    finally:
        db.close()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
