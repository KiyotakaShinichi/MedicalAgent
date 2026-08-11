from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.next_generation_evidence_reports import write_next_generation_reports


if __name__ == "__main__":
    paths = write_next_generation_reports()
    print(json.dumps({"reports": [path.relative_to(ROOT).as_posix() for path in paths]}, indent=2))
