# RAG Pipeline

MedicalAgent uses a deterministic-first RAG pipeline with optional LLM adjudication for routing and cache safety. The pipeline is designed for grounded knowledge support and portal help, not autonomous medical decision-making.

## Pipeline steps
1. Scope and safety check: Determine urgent, diagnosis, treatment-decision, or security boundary conditions before retrieval. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/services/security_guardrails.py](backend/services/security_guardrails.py)
2. Intent routing: Deterministic router with optional LLM adjudication for low-risk intents. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/services/local_llm.py](backend/services/local_llm.py)
3. Query rewrite and decomposition: Normalize and expand queries into subqueries. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)
4. Dense + sparse hybrid retrieval: Use local sentence-transformer embeddings + FAISS with BM25/RRF when dependencies are installed; otherwise fall back to sparse BM25 + TF-IDF with honest backend labels. Evidence: [backend/services/rag_vector_index.py](backend/services/rag_vector_index.py)
5. Parent-child context expansion: Expand related snippets for context continuity. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)
6. Reranking: Apply safety, domain, and source boosts for final ordering. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)
7. Contextual compression: Trim context to size-limited, high-signal snippets. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)
8. Citation-checked answer generation: Generate grounded answers and attach citations. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)
9. Validation and guardrails: Detect missing citations or unsafe treatment/diagnosis wording and refuse as needed. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)
10. Refusal and escalation: Urgent, unsafe, diagnosis/outcome, or treatment-decision requests are refused or escalated to clinician review. Patient-specific requests are handled with non-diagnostic language and are not eligible for caching. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/services/security_guardrails.py](backend/services/security_guardrails.py)

## Evidence pointers
- Core pipeline implementation: [backend/services/agent_rag.py](backend/services/agent_rag.py)
- Hybrid retrieval index: [backend/services/rag_vector_index.py](backend/services/rag_vector_index.py)
- Optional LLM adjudication: [backend/services/local_llm.py](backend/services/local_llm.py)
