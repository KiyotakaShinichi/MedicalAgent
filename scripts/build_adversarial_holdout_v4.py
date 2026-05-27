from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.adversarial_holdout_v4 import write_holdout_v4_bank  # noqa: E402


def main() -> int:
    cases = write_holdout_v4_bank()
    print(json.dumps({"cases": len(cases), "path": "Data/evals/safety/adversarial_holdout_v4.jsonl"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
