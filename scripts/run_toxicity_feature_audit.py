"""Run the toxicity feature-importance audit + no-proxy baseline."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.toxicity_feature_audit import run_toxicity_feature_audit


if __name__ == "__main__":
    payload = run_toxicity_feature_audit()
    baseline = payload.get("no_proxy_baseline", {})
    print(json.dumps({
        "status": payload.get("status"),
        "dominant_features": payload.get("dominant_features"),
        "near_label_proxy_features": payload.get("near_label_proxy_features"),
        "no_proxy_baseline_auc": baseline.get("auc"),
        "no_proxy_baseline_brier": baseline.get("brier"),
        "remaining_feature_count": baseline.get("remaining_feature_count"),
        "interpretation": payload.get("interpretation"),
    }, indent=2))
    # `needs_attention` is the honest default when a dominant feature
    # exists AND the no-proxy AUC stays high — surface but don't fail CI.
    sys.exit(0 if payload.get("status") in {"strong", "acceptable", "needs_attention"} else 1)
