"""Train the modality-robust treatment-response classifier and write the
model + metadata artifacts.  Exits non-zero if test metrics regress below
a sensible floor."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.modality_dropout_training import train_modality_robust_classifier


if __name__ == "__main__":
    metadata = train_modality_robust_classifier()
    test_metrics = metadata.get("test_metrics", {})
    print(json.dumps({
        "model_path": metadata.get("model_path"),
        "status": metadata.get("status"),
        "roc_auc": test_metrics.get("roc_auc"),
        "brier": test_metrics.get("brier"),
        "test_rows": test_metrics.get("test_rows"),
        "augmented_rows_added": metadata.get("augmentation_stats", {}).get("augmented_rows_added"),
        "mean_dropouts_per_augmented_row": metadata.get("augmentation_stats", {}).get("mean_dropouts_per_augmented_row"),
    }, indent=2))
    # Sanity floor: synthetic AUROC should still be well above chance even
    # with augmentation, otherwise something is wrong with the training data.
    auc = test_metrics.get("roc_auc") or 0.0
    sys.exit(0 if auc >= 0.7 else 1)
