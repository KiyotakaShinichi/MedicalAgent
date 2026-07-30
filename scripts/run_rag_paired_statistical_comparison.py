from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.rag_paired_statistical_comparison import (
    build_rag_paired_statistical_comparison,
)


def main() -> int:
    payload = build_rag_paired_statistical_comparison()
    print(json.dumps(payload["headline"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
