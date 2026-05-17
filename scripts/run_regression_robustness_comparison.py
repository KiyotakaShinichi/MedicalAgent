"""Run the legacy-vs-modality-robust regression comparison."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.regression_robustness_comparison import run_regression_robustness_comparison


if __name__ == "__main__":
    report = run_regression_robustness_comparison()
    summary = report.get("summary", {})
    print(json.dumps({
        "status": report.get("status"),
        "robust_wins": summary.get("force_score_mae_wins_for_robust"),
        "robust_losses": summary.get("force_score_mae_losses_for_robust"),
        "full_data_mae_delta": summary.get("full_data_mae_delta"),
        "scenario_count": summary.get("scenario_count"),
    }, indent=2))
    sys.exit(0 if report.get("status") in {"robust", "acceptable"} else 1)
