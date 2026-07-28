from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.xai_rank_stability_audit import build_xai_rank_stability_audit


def main() -> int:
    payload = build_xai_rank_stability_audit()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "patient_explanation_n": payload["patient_explanation_n"],
                "bootstrap_n": payload["bootstrap_n"],
                "grouped_top_k_jaccard_p05": payload[
                    "patient_display_grouped_ranking"
                ]["top_k_jaccard_p05"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
