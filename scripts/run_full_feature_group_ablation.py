from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.full_feature_group_ablation import run_full_feature_group_ablation


if __name__ == "__main__":
    report = run_full_feature_group_ablation()
    print(json.dumps({
        "status": report["status"],
        "feature_group_count": len(report["feature_groups"]),
        "recommended_use": report["recommendation"]["recommended_use"],
        "full_vs_clinical_auroc_delta": report["deltas"]["full_vs_clinical_auroc_delta"],
        "full_vs_clinical_brier_delta": report["deltas"]["full_vs_clinical_brier_delta"],
    }, indent=2))
