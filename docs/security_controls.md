# Security Controls (PoC)

This document describes current security controls and what is still required for production deployment.

## Implemented controls
- Deterministic guardrails for prompt injection and privacy boundary
- Role-based routing and access context checks
- Audit logging for model and application events
- PII redaction for logged payloads and query previews
- Cache policy blocking patient-specific and unsafe content
- Basic request tracing with request IDs

## Controls still needed for production
- Strong authentication (MFA, password policies)
- Principle-of-least-privilege authorization with reviews
- Encryption at rest and in transit
- Secrets management and rotation
- Formal data retention and deletion policy
- PHI redaction and access monitoring at scale
