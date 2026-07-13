# Pinecone Shadow Retrieval Comparison

Pinecone shadow retrieval is an optional managed-vector-search comparison scaffold. It is disabled by default, must not store PHI or raw patient chat, and does not replace source-tier filtering, allowed-use filtering, citation validation, safety refusal, or local FAISS/BM25 fallback. This artifact is not clinical validation, not healthcare production readiness, and not proof of retrieval improvement until a real shadow run is completed on frozen evals.

## Current Status

- Status: `ready_for_shadow_mode_not_configured`
- Pinecone configured: `False`
- Comparison completed: `False`
- PHI allowed: `False`
- Live patient route enabled: `False`

## Local Reference

- Full-stack Recall@10: `0.7838`
- Full-stack citation precision: `0.5243`
- Full-stack unsupported context rate: `0.1622`
- Full-stack source-tier correctness: `1.0`

## Metadata Filter Contract

- `source_tier`
- `allowed_use`
- `patient_facing`
- `staleness_status`
- `kb_fingerprint`
- `clinical_validation=false`
- `doc_type`

## Promotion Gate

- no PHI or patient-specific memory in Pinecone
- source_tier_correctness remains 1.0
- unsafe_answer_rate remains 0.0 in live-agent eval
- citation_precision does not regress versus local full stack
- unsupported_context_rate does not increase versus local full stack
- latency/cost tradeoff is explicitly reported
- local FAISS/BM25 fallback remains available

## Blocked Claims

- clinical validation
- retrieval improvement proven
- production healthcare readiness
- HIPAA compliance
- real patient safety
- patient-facing clinical confidence
