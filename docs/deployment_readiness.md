# Deployment Readiness

## Authentication boundary

The backend now includes a feature-flagged OIDC/JWKS bearer adapter. It accepts
only RS256 tokens and validates signature, issuer, audience, expiration,
issued-at time, subject, role mapping, and patient scope. Strict profiles remain
blocked unless the issuer, audience, HTTPS JWKS endpoint, and role claims are
explicitly configured.

This is an API authentication adapter, not a complete identity program. Browser
authorization-code/PKCE login, identity proofing, provider-side logout and
revocation, MFA policy, account lifecycle, consent, audit review, and formal
security/compliance assessment are still absent.

Optional external automation dispatch also fails preflight unless it uses
HTTPS, a non-placeholder signing secret of at least 32 characters, and
synthetic test-recipient mode.

NLCare can be packaged and smoke-tested like a deployment-shaped engineering
prototype, but it is **not** healthcare-production-ready.

It still has no clinical validation, no clinician sign-off, no real patient
cohort, no IRB/ethics approval, and no formal PHI/compliance review.

## Preflight

Run:

```bash
python scripts/run_deployment_readiness.py
python scripts/run_container_recovery_smoke.py
python scripts/ship.py
```

The deployment preflight writes:

```text
Data/evals/ops/latest_deployment_readiness.json
```

The artifact checks environment posture, demo-auth risk, CORS configuration,
Docker assets, release-gate availability, and the production-readiness boundary.

## Runtime Probes

FastAPI exposes:

```text
GET /health
GET /ready
```

`/ready` verifies database reachability and returns explicit boundary fields:

- `clinical_validation: false`
- `healthcare_production_ready: false`
- `demo_auth_allowed`
- `claim_boundary`

## Production-Shaped Local Compose

For a local deployment-shaped smoke test:

```bash
docker compose -f docker-compose.prod.yml up --build
```

Frontend:

```text
http://127.0.0.1:8080
```

Backend:

```text
http://127.0.0.1:8017
```

This Compose file uses a static Vite build served by Nginx and proxies `/api`
to the FastAPI service. It is still a local/staging-style setup, not PHI-ready
or hospital-ready infrastructure.

## Recovery Evidence

`python scripts/run_deployment_recovery_drill.py` exercises a temporary SQLite
backup and exact-content restore over synthetic operational rows. The artifact
checks database integrity, restored row count, and a canonical content hash.

This is deliberately local-only. Managed PostgreSQL point-in-time recovery,
encrypted retention, multi-instance restoration, and approved production RPO
and RTO remain untested and required before any production claim.

`python scripts/run_container_recovery_smoke.py` is the stricter disposable
container check. When Docker is available it applies Alembic migrations to a
fresh Postgres 16 database, inserts a synthetic marker, performs `pg_dump` and
`pg_restore` into a second database, compares migration/table/marker state,
restarts Redis 7 with AOF persistence, verifies the marker, and removes only the
isolated smoke project's volumes. If Docker is unavailable or stuck, the
artifact records `blocked_environment` and `completed: false`.

## Required Before Any Real Healthcare Deployment

- Complete browser OIDC authorization-code/PKCE integration and provider-side
  session lifecycle around the API bearer adapter.
- Complete external-author RAG/adversarial evals.
- Complete nurse/clinician safety wording review.
- Complete genetic counselor review for genetics/VUS behavior.
- Complete senior MLE review of eval design.
- Add real or externally reviewed data with exact label mapping.
- Complete IRB/ethics governance for any real patient data.
- Complete PHI/security/compliance review.
- Add hosted monitoring, incident response, managed-database backups, tested
  point-in-time recovery, and rollback procedures.

## Claim Boundary

Good claim:

> Deployment-shaped engineering prototype with preflight checks, health probes,
> release gates, and explicit clinical boundaries.

Blocked claims:

- production healthcare deployment
- clinically validated system
- safe for real patient care
- PHI-ready or HIPAA-compliant deployment
- hospital/EHR-ready integration
