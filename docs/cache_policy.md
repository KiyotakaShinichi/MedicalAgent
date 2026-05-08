# Cache Policy

MedicalAgent caches only low-risk, reusable educational or portal-help answers that are not patient-specific. All patient-specific, urgent, treatment-decision, or privacy-sensitive content is blocked from caching.

## Cache types
- Exact cache: Matches normalized queries to validated responses. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)
- Semantic cache: Reuses low-risk answers when semantic-key similarity is high. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)

## Safety controls
- TTL expiration: Cached answers expire after a fixed number of days. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)
- Knowledge-base fingerprint invalidation: Cached answers are invalidated when the KB fingerprint changes. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/services/rag_vector_index.py](backend/services/rag_vector_index.py)
- Cache schema versioning: Cache entries include a schema version for compatibility checks. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py), [backend/models.py](backend/models.py)
- Safety policy rejection: Cache storage is rejected for urgent or patient-specific requests. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)

## Content that must never be cached
- Patient-specific content
- Urgent medical requests
- Treatment-decision content
- Unsafe medical advice
- Privacy-sensitive content

## Content that may be cached
- Low-risk general educational answers that are not patient-specific
- Portal help and navigation guidance that are not patient-specific

## Citation requirement
- If retrieval context is used, the cached answer must include citations.
- Direct fallback answers without retrieved context may be cached only when they remain low-risk and non-patient-specific.

## Implementation references
- Cache checks and storage: [backend/services/agent_rag.py](backend/services/agent_rag.py)
- Cache persistence model: [backend/models.py](backend/models.py)
