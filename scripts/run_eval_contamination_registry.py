from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.eval_contamination_registry import run_eval_contamination_registry  # noqa: E402


def main() -> int:
    payload = run_eval_contamination_registry()
    print(json.dumps({
        "status": payload["status"],
        "registry_entry_count": payload["registry_entry_count"],
        "used_for_tuning_entry_count": payload["used_for_tuning_entry_count"],
        "external_authored_entry_count": payload["external_authored_entry_count"],
        "frozen_or_holdout_entry_count": payload["frozen_or_holdout_entry_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
