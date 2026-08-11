# NLCare Tenant-Isolation Security

Generated from repository artifacts at `2026-08-11T06:52:48.687832+00:00`.

> NLCare remains synthetic-only, non-diagnostic, not clinically validated, and not production healthcare ready. Internal tests are engineering evidence, not evidence of patient benefit or medical effectiveness.

## Controlled matrix

Three disposable synthetic tenants exercise foreign IDs, role confusion, cache keys, vector namespaces, project relationships, worker scope, and authorization bypass attempts.

- Status: `passed`.
- Cases: `NOT_RUN`.
- Passed: `NOT_RUN`.
- Leakage rate: `NOT_RUN`.
- This is controlled local security regression evidence, not a penetration test or certification.

## Remaining deployment work

- Managed PostgreSQL row-level policy verification.
- Real OIDC provider integration and session revocation drills.
- Gateway/WAF configuration, secret rotation, independent penetration testing, and incident ownership.
