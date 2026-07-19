import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.synthetic_v2_model_stability import run_synthetic_v2_model_stability


if __name__ == "__main__":
    result = run_synthetic_v2_model_stability()
    print(f"status={result['status']} runs={len(result['runs'])} decision={result['promotion_decision']['decision']}")
