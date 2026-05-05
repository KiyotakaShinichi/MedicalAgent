# Knowledge Base Staging

Put future breast cancer research papers, guideline notes, and patient-education source files in:

```text
KnowledgeBase/raw/
```

Supported starter formats:

- `.md`
- `.txt`
- `.pdf` if `pypdf` is installed

Then run:

```text
python scripts/ingest_knowledge_base.py
```

The ingestion script writes local RAG chunks to:

```text
Data/rag_knowledge_base_chunks.json
```

That output is intentionally ignored by git because source documents may have licensing restrictions and may contain large files.
