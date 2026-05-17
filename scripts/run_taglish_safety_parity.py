"""Run the Taglish ↔ English safety-route parity check."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.agent_rag import route_intent
from backend.services.taglish_safety_parity import run_parity_check


def _safety_detector(query: str) -> dict:
    # Lazy-import so the script doesn't crash when optional safety deps
    # are missing in fresh checkouts.
    try:
        from backend.processing.safety_scope import detect_safety_scope
        return detect_safety_scope(query) or {}
    except Exception:
        return {}


def _intent_router(query: str, safety: dict) -> str:
    return route_intent(query, actions=None, safety=safety)


if __name__ == "__main__":
    payload = run_parity_check(
        safety_detector=_safety_detector,
        intent_router=_intent_router,
    )
    print(json.dumps({
        "status": payload.get("status"),
        "case_count": payload.get("case_count"),
        "passed": payload.get("passed"),
        "pass_rate": payload.get("pass_rate"),
        "intent_parity_rate": payload.get("intent_parity_rate"),
        "safety_scope_parity_rate": payload.get("safety_scope_parity_rate"),
    }, indent=2))
    sys.exit(0 if payload.get("status") in {"strong", "acceptable"} else 1)
