from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.adversarial_v6_retrospective import write_retrospective


if __name__ == "__main__":
    result = write_retrospective()
    print(
        f"status={result['status']} failures={result['failed_n']} "
        f"tuning_used={result['was_used_for_tuning']}"
    )
