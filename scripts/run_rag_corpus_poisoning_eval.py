import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.rag_corpus_poisoning_eval import build_corpus_poisoning_eval  # noqa: E402


if __name__ == "__main__":
    report = build_corpus_poisoning_eval()
    print(
        f"corpus poisoning: {report['status']} "
        f"({report['passed_count']}/{report['case_count']}, "
        f"generation_context_poison_rate={report['generation_context_poison_rate']})"
    )
