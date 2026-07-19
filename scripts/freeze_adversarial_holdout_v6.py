import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.adversarial_holdout_v6 import freeze_holdout_v6


if __name__ == "__main__":
    result = freeze_holdout_v6()
    print(f"frozen={result['total_n']} sha256={result['sha256']}")
