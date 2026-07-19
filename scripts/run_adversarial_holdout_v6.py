import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.adversarial_holdout_v6 import evaluate_holdout_v6


if __name__ == "__main__":
    result = evaluate_holdout_v6()
    print(
        f"status={result['status']} pass_rate={result['pass_rate']} "
        f"unsafe_leakage_rate={result['unsafe_leakage_rate']} "
        f"over_refusal_rate={result['over_refusal_rate']}"
    )
