import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.unsafe_intent_mutation_dev_eval import evaluate_mutation_dev


if __name__ == "__main__":
    result = evaluate_mutation_dev()
    print(f"mutation dev pass_rate={result['pass_rate']:.4f} safe_negative={result['safe_negative_pass_rate']:.4f}")
