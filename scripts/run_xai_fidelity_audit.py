import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.xai_fidelity_audit import build_xai_fidelity_audit


if __name__ == "__main__":
    result = build_xai_fidelity_audit()
    print(f"status={result['status']} additivity_verifiable={result['additivity_verifiable']}")
