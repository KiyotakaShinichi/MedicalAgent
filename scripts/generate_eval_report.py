import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.database import SessionLocal  # noqa: E402
from backend.services.evaluation_reports import generate_versioned_evaluation_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate a versioned evaluation report.")
    parser.add_argument("--output-root", default="Data/model_evaluation_reports")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = generate_versioned_evaluation_report(
            db=db,
            output_root=args.output_root,
            run_id=args.run_id,
        )
    finally:
        db.close()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
