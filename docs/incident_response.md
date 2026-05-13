# Incident Response (PoC)

This is a lightweight incident response outline for the PoC. It is not a full compliance program.

## Severity levels
- Sev1: Data exposure, cross-patient access, or unsafe medical advice in production
- Sev2: Repeated guardrail failures or abnormal safety regressions
- Sev3: Non-critical errors, performance degradation, or partial outages

## Triage steps
1. Identify the scope (routes, patients, or logs impacted)
2. Stop the bleeding (disable endpoints, block cache, or roll back model)
3. Preserve evidence (audit logs, request IDs, evaluation reports)
4. Fix the root cause and add a regression test

## Communication
- Notify internal stakeholders for Sev1/Sev2
- Document timeline, impact, and resolution steps

## Post-incident actions
- Add regression cases for the failure
- Review guardrail thresholds and update runbooks
- Re-evaluate model or RAG caches if needed
