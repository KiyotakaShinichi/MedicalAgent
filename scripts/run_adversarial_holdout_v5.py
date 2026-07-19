import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.adversarial_holdout_v5 import evaluate_holdout_v5


if __name__ == "__main__":
    result = evaluate_holdout_v5()
    print(f"v5 pass_rate={result['pass_rate']:.4f} unsafe_leakage={result['unsafe_leakage_rate']:.4f}")
