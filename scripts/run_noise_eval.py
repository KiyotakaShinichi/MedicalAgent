"""Standalone script: run high-noise synthetic evaluation and export JSON report."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.noise_eval import run_noise_eval

if __name__ == "__main__":
    result = run_noise_eval()
    print(json.dumps({
        "output_path": result.get("noise_eval_report", "Data/mle_monitoring/noise_eval_report.json"),
        "status": result.get("status"),
        "baseline_auroc": (result.get("baseline") or {}).get("auroc"),
        "max_auroc_drop": result.get("max_auroc_drop"),
        "noise_results": {
            k: v.get("auroc_drop") if isinstance(v, dict) else None
            for k, v in (result.get("noise_scenarios") or {}).items()
        },
        "interpretation": result.get("interpretation"),
    }, indent=2))
