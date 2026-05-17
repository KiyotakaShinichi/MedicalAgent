"""Build the KB source-governance artifact (tier + allowed_use + staleness)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.kb_source_governance import build_kb_source_governance


if __name__ == "__main__":
    payload = build_kb_source_governance()
    print(json.dumps({
        "status": payload.get("status"),
        "source_count": payload.get("source_count"),
        "tier_distribution": payload.get("tier_distribution"),
        "allowed_use_distribution": payload.get("allowed_use_distribution"),
        "staleness_distribution": payload.get("staleness_distribution"),
        "issue_count": len(payload.get("governance_issues", [])),
    }, indent=2))
    # Governance reports `needs_attention` honestly when uncategorised
    # trust_levels exist — this is information, not a CI failure unless the
    # KB cannot be loaded at all.
    sys.exit(0 if payload.get("status") in {"strong", "acceptable", "needs_attention"} else 1)
