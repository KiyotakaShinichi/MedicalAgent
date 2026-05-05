# Knowledge Base Staging

Put future breast cancer research papers, guideline notes, and patient-education source files in:

```text
KnowledgeBase/raw/
```

Supported starter formats:

- `.md`
- `.txt`
- `.pdf` if `pypdf` is installed

To download the starter open-access breast cancer research set:

```text
python scripts/download_research_papers.py
```

The downloader uses the NCBI Open Access subset and writes full-text `.txt`
files into:

```text
KnowledgeBase/raw/research_papers/
```

Direct PDF links from PubMed Central can be gated or redirected in some
environments, so the project stores clean open-access article text for RAG
instead of relying on browser-only PDF downloads.

Then run:

```text
python scripts/ingest_knowledge_base.py
```

After ingestion, run the agent regression suite:

```text
python scripts/evaluate_agent_rag.py
```

This checks whether the updated KB still preserves expected retrieval, citations, grounding proxies, and prompt-injection/privacy blocking.

The ingestion script writes local RAG chunks to:

```text
Data/rag_knowledge_base_chunks.json
```

That output is intentionally ignored by git because source documents may have licensing restrictions and may contain large files.

Starter paper notes live in:

```text
KnowledgeBase/STARTER_PAPERS.md
```
