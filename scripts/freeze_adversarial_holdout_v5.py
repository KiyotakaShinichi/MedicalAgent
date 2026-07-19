import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.adversarial_holdout_v5 import freeze_holdout_v5


if __name__ == "__main__":
    manifest = freeze_holdout_v5()
    print(f"frozen adversarial v5: n={manifest['total_n']} sha256={manifest['sha256']}")
