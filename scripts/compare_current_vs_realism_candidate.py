import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.candidate_model_comparison import build_current_vs_candidate_report  # noqa: E402


def main():
    report = build_current_vs_candidate_report()
    print(json.dumps({
        "output_path": "Data/mle_monitoring/current_vs_realism_candidate.json",
        "current": report.get("current"),
        "candidate": report.get("candidate"),
        "recommendation": report.get("recommendation"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
