import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.demo_storyline import build_demo_storyline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate a repeatable demo storyline from local DB records.")
    parser.add_argument("--patient-id", default="P001")
    args = parser.parse_args()
    report = build_demo_storyline(patient_id=args.patient_id)
    print(json.dumps({
        "patient_id": args.patient_id,
        "files": report.get("files"),
        "scene_count": len(report.get("scenes") or []),
    }, indent=2))


if __name__ == "__main__":
    main()
