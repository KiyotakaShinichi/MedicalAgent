import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.agent_rag import get_rag_corpus, knowledge_base_fingerprint  # noqa: E402
from backend.services.rag_vector_index import DEFAULT_RAG_INDEX_PATH, build_rag_vector_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build the local hybrid RAG retrieval index.")
    parser.add_argument("--index-path", default=DEFAULT_RAG_INDEX_PATH)
    args = parser.parse_args()

    corpus = get_rag_corpus()
    result = build_rag_vector_index(
        corpus=corpus,
        index_path=args.index_path,
        knowledge_fingerprint=knowledge_base_fingerprint(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
