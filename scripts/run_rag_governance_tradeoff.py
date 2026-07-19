import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.rag_governance_tradeoff import write_rag_governance_tradeoff


if __name__ == "__main__":
    artifact = write_rag_governance_tradeoff()
    print(
        "RAG governance tradeoff:",
        artifact["status"],
        artifact["tradeoffs"],
        "external holdout completed=", artifact["external_holdout"]["completed"],
    )
