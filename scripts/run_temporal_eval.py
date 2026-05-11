"""Standalone script: run temporal train/eval split and export JSON report."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.temporal_eval import run_temporal_eval

if __name__ == "__main__":
    result = run_temporal_eval()
    print(json.dumps({
        "output_path": result.get("temporal_eval_report", "Data/mle_monitoring/temporal_eval_report.json"),
        "status": result.get("status"),
        "temporal_auroc": (result.get("temporal_split") or {}).get("eval_auroc"),
        "baseline_auroc": (result.get("random_split_baseline") or {}).get("eval_auroc"),
        "generalization_gap": result.get("generalization_gap"),
        "interpretation": result.get("interpretation"),
    }, indent=2))
