# Environment configuration

`.env.example` is the canonical human-readable runtime configuration inventory.
Every application variable referenced through `os.getenv` or `os.environ` in
`backend/` and `scripts/` must appear there. CI enforces this with
`python scripts/check_env_documentation.py`.

## Deployment rules

- Copy `.env.example` only as a local starting point. Never commit `.env`.
- `GROQ_API_KEY`, `PINECONE_API_KEY`, `AZURE_SEARCH_API_KEY`, bearer tokens,
  database credentials, and webhook signing secrets must come from a secret
  manager in deployed environments.
- Keep `NLCARE_SYNTHETIC_ONLY=true`. This repository has no approval for real
  patient data or patient care.
- Keep experimental agentic, fine-tuning, managed-vector, and automation
  execution switches disabled unless a bounded synthetic evaluation explicitly
  requires them.
- The `ONCOTRACK_*` variables are compatibility aliases only. New deployments
  should use `NLCARE_*` names.
- DEP-001 artifact paths are protocol-owned. Ordinary development must not use
  them to inspect, rewrite, or rerun consumed one-shot banks.

## Environments

`APP_ENV`/`ENVIRONMENT` should be one of `development`, `test`, `staging`, or
`production`. Staging and production must disable demo authentication, use a
durable database, provide Redis when shared rate limiting is enabled, and pass
`/ready` before receiving traffic. Passing `/ready` demonstrates bounded
engineering dependency availability only; it does not establish clinical
validation, healthcare compliance, or real-patient readiness.
