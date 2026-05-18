from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.services.hybrid_subgroup_metrics import run_hybrid_subgroup_metrics


if __name__ == "__main__":
    r = run_hybrid_subgroup_metrics()
    print(json.dumps({"status": r["status"], "rows": r["overall"]["n"]}, indent=2))
