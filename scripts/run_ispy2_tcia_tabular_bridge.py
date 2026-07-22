"""Run the isolated TCIA I-SPY2 tabular stress benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.ispy2_tcia_tabular_bridge import run_ispy2_tcia_external_stress


if __name__ == "__main__":
    report = run_ispy2_tcia_external_stress()
    print(json.dumps({
        "status": report["status"],
        "joined_row_count": report["source"]["joined_row_count"],
        "used_for_nlcare_training": report["used_for_nlcare_training"],
        "promotion_allowed": report["promotion_allowed"],
    }, indent=2))
