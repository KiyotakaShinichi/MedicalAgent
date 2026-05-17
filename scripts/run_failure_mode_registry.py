"""Build the consolidated failure-mode registry."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.failure_mode_registry import build_failure_mode_registry


if __name__ == "__main__":
    payload = build_failure_mode_registry()
    summary = payload.get("summary", {})
    print(json.dumps({
        "status": payload.get("status"),
        "entry_count": payload.get("entry_count"),
        "by_severity": summary.get("by_severity"),
        "entries_with_unresolved_gap": summary.get("entries_with_unresolved_gap"),
    }, indent=2))
    # `needs_attention` is the honest default — the registry's job is to
    # document unresolved gaps, not to be empty.  CI only fails when the
    # registry can't be built or parsed at all.
    sys.exit(0 if payload.get("status") in {"strong", "acceptable", "needs_attention"} else 1)
