from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import SessionLocal
from backend.schema_migrations import ensure_schema
from backend.services.demo_patient_sync import sync_demo_patient_journey


def main():
    ensure_schema()
    db = SessionLocal()
    try:
        result = sync_demo_patient_journey(db)
        print(result)
    finally:
        db.close()


if __name__ == "__main__":
    main()
