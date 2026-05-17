"""Train the modality-robust regression head + write the metadata artifact."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.modality_dropout_regression_training import train_modality_robust_regressor


if __name__ == "__main__":
    metadata = train_modality_robust_regressor()
    test = metadata.get("test_metrics", {})
    aug = metadata.get("augmentation_stats", {})
    print(json.dumps({
        "status": metadata.get("status"),
        "model_path": metadata.get("model_path"),
        "test_mae": test.get("mae"),
        "test_rmse": test.get("rmse"),
        "test_rows": test.get("test_rows"),
        "augmented_rows_added": aug.get("augmented_rows_added"),
        "mean_dropouts_per_augmented_row": aug.get("mean_dropouts_per_augmented_row"),
    }, indent=2))
    sys.exit(0 if metadata.get("status") in {"strong", "acceptable"} else 1)
