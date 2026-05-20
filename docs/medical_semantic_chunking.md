# Medical Semantic Chunking

NLCare now includes a structure-aware semantic chunking scaffold for medical knowledge-base documents.

This improves retrieval provenance and context integrity. It is not clinical validation and does not make the answer medically correct by itself.

## What It Preserves

- Markdown heading hierarchy as chunk metadata.
- Parent-child section linkage.
- Source ID, chunk ID, source tier, staleness, and allowed-use metadata.
- Lab/date, imaging findings/impression, medication timing, and family-history relationship context where possible.

## Evaluation

Run:

```bash
python scripts/rebuild_kb_with_semantic_chunking.py
```

Artifact:

```text
Data/evals/rag/latest_chunking_quality_eval.json
```

Key metrics:

- `heading_metadata_coverage`
- `parent_child_link_coverage`
- `critical_context_split_rate`
- `chunk_source_traceability`

The current script writes a preview/evaluation artifact. Replacing the live KB index should remain an explicit release decision.
