from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.oidc_pkce import build_oidc_browser_pkce_readiness


if __name__ == "__main__":
    report = build_oidc_browser_pkce_readiness()
    print(json.dumps({"status": report["status"], "production_auth_ready": report["production_auth_ready"]}, indent=2))
