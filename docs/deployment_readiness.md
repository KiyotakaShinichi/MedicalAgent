# Deployment Readiness

NLCare can be packaged and smoke-tested like a deployment-shaped engineering
prototype, but it is **not** healthcare-production-ready.

It still has no clinical validation, no clinician sign-off, no real patient
cohort, no IRB/ethics approval, and no formal PHI/compliance review.

## Preflight

Run:

```bash
python scripts/run_deployment_readiness.py
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

## Required Before Any Real Healthcare Deployment

- Replace demo authentication with real auth.
- Complete external-author RAG/adversarial evals.
- Complete nurse/clinician safety wording review.
- Complete genetic counselor review for genetics/VUS behavior.
- Complete senior MLE review of eval design.
- Add real or externally reviewed data with exact label mapping.
- Complete IRB/ethics governance for any real patient data.
- Complete PHI/security/compliance review.
- Add hosted monitoring, incident response, backups, and rollback procedures.

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
