from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001d_blind_bank import build_and_freeze_dep001d_blind_bank


if __name__ == "__main__":
    manifest = build_and_freeze_dep001d_blind_bank()
    print(json.dumps({
        "blind_bank_id": manifest["blind_bank_id"],
        "case_n": manifest["case_n"],
        "frozen": manifest["frozen"],
        "bank_sha256": manifest["blind_bank_sha256"],
    }, indent=2))
