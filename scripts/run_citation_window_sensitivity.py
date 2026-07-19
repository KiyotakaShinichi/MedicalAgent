from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.rag_citation_window_sensitivity import build_citation_window_sensitivity  # noqa: E402


def main() -> int:
    report = build_citation_window_sensitivity()
    print(
        "citation_window_sensitivity "
        f"status={report['status']} "
        f"recommended_k={report['recommended_cited_context_k']} "
        f"promotion={report['promotion_recommendation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
