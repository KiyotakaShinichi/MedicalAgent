from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.claim_conditioned_citation_selector_eval import (
    build_claim_conditioned_citation_selector_eval,  # noqa: E402
)


if __name__ == "__main__":
    report = build_claim_conditioned_citation_selector_eval()
    print(
        f"status={report['status']} precision_delta={report['citation_precision_delta']} "
        f"promotion={report['promotion_decision']}"
    )
