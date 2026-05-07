import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from backend.services.kb_ingestion import clear_ingested_chunk_cache, ingest_knowledge_base
from backend.services.agent_rag import get_rag_corpus, knowledge_base_fingerprint
from backend.services.rag_vector_index import DEFAULT_RAG_INDEX_PATH, build_rag_vector_index


def main():
    parser = argparse.ArgumentParser(description="Ingest local RAG knowledge-base files into chunk JSON.")
    parser.add_argument("--input-dir", default="KnowledgeBase/raw")
    parser.add_argument("--output-path", default="Data/rag_knowledge_base_chunks.json")
    parser.add_argument("--chunk-chars", type=int, default=2200)
    parser.add_argument("--overlap-chars", type=int, default=220)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--index-path", default=DEFAULT_RAG_INDEX_PATH)
    args = parser.parse_args()

    result = ingest_knowledge_base(
        input_dir=args.input_dir,
        output_path=args.output_path,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )
    clear_ingested_chunk_cache()
    if not args.skip_index:
        result["rag_index"] = build_rag_vector_index(
            corpus=get_rag_corpus(),
            index_path=args.index_path,
            knowledge_fingerprint=knowledge_base_fingerprint(),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
