from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.medical_semantic_chunker import evaluate_chunking_quality, load_markdown_documents


KB_ROOT = Path("KnowledgeBase/raw/curated_medical_kb")
OUTPUT_PATH = Path("Data/evals/rag/latest_chunking_quality_eval.json")
PREVIEW_PATH = Path("Data/rag_index/semantic_chunk_preview.json")


def main() -> None:
    documents = load_markdown_documents(KB_ROOT)
    if not documents:
        documents = [(
            "fallback_semantic_chunk_sample",
            "# CBC monitoring\n\nWBC collected 2026-01-01: 5.2 10^9/L.\n\n# Imaging\n\nFindings: synthetic text.\n\nImpression: synthetic context.",
            {"allowed_use": ["education"], "source_tier": "T3", "staleness": "unknown"},
        )]
    payload = evaluate_chunking_quality(documents)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    preview = {
        "schema_version": "semantic_chunk_preview_v1",
        "source": str(KB_ROOT),
        "note": "Preview artifact only; live index replacement remains explicit.",
        "sample_document_count": len(documents),
    }
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_PATH.write_text(json.dumps(preview, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "chunk_count": payload["chunk_count"],
        "critical_context_split_rate": payload["critical_context_split_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
