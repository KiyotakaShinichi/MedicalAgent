import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from backend.services.kb_ingestion import ingest_knowledge_base


def main():
    parser = argparse.ArgumentParser(description="Ingest local RAG knowledge-base files into chunk JSON.")
    parser.add_argument("--input-dir", default="KnowledgeBase/raw")
    parser.add_argument("--output-path", default="Data/rag_knowledge_base_chunks.json")
    parser.add_argument("--chunk-chars", type=int, default=1400)
    parser.add_argument("--overlap-chars", type=int, default=180)
    args = parser.parse_args()

    result = ingest_knowledge_base(
        input_dir=args.input_dir,
        output_path=args.output_path,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
