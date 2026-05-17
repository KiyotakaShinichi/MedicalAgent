"""Run the champion-vs-robust comparison and write the artifact."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.modality_robustness_comparison import (
    run_modality_robustness_comparison,
)


if __name__ == "__main__":
    payload = run_modality_robustness_comparison()
    summary = payload.get("summary", {})
    print(json.dumps({
        "status": payload.get("status"),
        "scenario_count": summary.get("scenario_count"),
        "robust_wins": summary.get("force_score_accuracy_wins_for_robust"),
        "robust_losses": summary.get("force_score_accuracy_losses_for_robust"),
        "full_data_accuracy_delta": summary.get("full_data_accuracy_delta"),
        "full_data_brier_delta": summary.get("full_data_brier_delta"),
    }, indent=2))
    sys.exit(0 if payload.get("status") in {"robust", "acceptable"} else 1)
