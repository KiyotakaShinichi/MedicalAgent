"""Train modality-dropout p10/p50/p90 response-score quantile heads."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.modality_dropout_quantile_regression_training import (
    train_modality_dropout_quantile_regression_heads,
)


if __name__ == "__main__":
    metadata = train_modality_dropout_quantile_regression_heads()
    interval = metadata.get("interval", {})
    comparison = metadata.get("scenario_comparison", {})
    print(json.dumps({
        "status": metadata.get("status"),
        "artifact_paths": metadata.get("artifact_paths"),
        "nominal_coverage": interval.get("nominal_coverage"),
        "empirical_coverage": interval.get("empirical_coverage"),
        "median_band_width": interval.get("median_band_width"),
        "robust_mae_wins": comparison.get("robust_mae_wins"),
        "robust_mae_losses": comparison.get("robust_mae_losses"),
        "test_rows": metadata.get("test_rows"),
    }, indent=2))
    sys.exit(0 if metadata.get("status") in {"strong", "acceptable"} else 1)
