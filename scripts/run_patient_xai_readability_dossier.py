from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.patient_xai_readability_dossier import build_patient_xai_readability_dossier  # noqa: E402


def main() -> int:
    report = build_patient_xai_readability_dossier()
    print(json.dumps({
        "status": report["status"],
        "surface_count": report["surface_count"],
        "failed_check_count": report["failed_check_count"],
        "rag_citation_precision": report["weakness_visibility"]["rag"].get("citation_precision"),
        "ml_attention_items": report["weakness_visibility"]["ml"].get("known_attention_items"),
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
