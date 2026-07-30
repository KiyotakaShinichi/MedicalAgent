from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.finetune_semantic_contamination import (
    build_finetune_semantic_contamination_audit,
)


def main() -> int:
    payload = build_finetune_semantic_contamination_audit()
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
