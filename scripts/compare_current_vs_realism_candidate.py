import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.realism_candidate_ab_gate import run_realism_candidate_ab_gate  # noqa: E402


def main():
    report = run_realism_candidate_ab_gate()
    print(json.dumps({
        "output_path": "Data/mle_monitoring/current_vs_realism_candidate.json",
        "status": report.get("status"),
        "deltas": report.get("deltas"),
        "recommendation": report.get("recommendation"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
