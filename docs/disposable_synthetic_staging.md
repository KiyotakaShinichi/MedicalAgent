# Disposable synthetic staging

`docker-compose.synthetic-staging.yml` defines a loopback-only environment for
non-patient engineering drills:

- PostgreSQL and Redis
- FastAPI backend and background worker
- React development frontend
- n8n with the workflow directory mounted read-only
- MailHog for local-only email capture
- managed-vector network and shadow modes disabled by default

All published ports bind to `127.0.0.1`. Automation dispatch is disabled until
an operator deliberately imports the inactive workflow and runs a signed
synthetic event drill. The compose file uses only local synthetic credentials;
it must not be reused as a shared or healthcare-production configuration.

Run:

```bash
python scripts/run_disposable_synthetic_staging_readiness.py
docker compose -f docker-compose.synthetic-staging.yml up --build
```

The readiness artifact records static validation when Docker is unavailable.
It does not claim the stack ran, that an email reached a real person, that a
managed vector provider was called, or that the system is clinically validated
or production healthcare ready.
