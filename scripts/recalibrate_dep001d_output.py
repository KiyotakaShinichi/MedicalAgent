from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001d_training import recalibrate_dep001d_output_thresholds


if __name__ == "__main__":
    result = recalibrate_dep001d_output_thresholds()
    print(f"DEP-001D output recalibration: {result['status']}")
    print(result["thresholds"]["output"])
    print(result["output_actionability"]["internal_test"])
