import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.genetic_counseling_eval import run_genetic_counseling_eval


def main():
    report = run_genetic_counseling_eval()
    print(json.dumps({
        "status": report["status"],
        "case_count": report["case_count"],
        "metrics": report["metrics"],
        "output_path": "Data/evals/genetics/latest_genetic_counseling_eval.json",
    }, indent=2))


if __name__ == "__main__":
    main()
