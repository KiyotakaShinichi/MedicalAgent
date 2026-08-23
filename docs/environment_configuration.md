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

## Runtime probes

Two endpoints, answering two different questions. Conflating them is the
mistake this split exists to avoid.

### `GET /health` — liveness

Returns **200 whenever the process is alive**, and a typed JSON body:

```json
{
  "status": "ok",
  "service": "nlcare_monitoring_prototype",
  "version": "0.0.0",
  "database": { "connected": true, "error_type": null },
  "rag_index": { "loaded": false, "error_type": null }
}
```

| field | meaning |
| --- | --- |
| `status` | `ok` whenever the process can serve requests. Never tracks a dependency. |
| `service` | Stable service identifier. |
| `version` | Running build, or `unknown` if it could not be resolved. Tells an operator *which* build answered. |
| `database.connected` | Whether a bounded `SELECT 1` succeeded at probe time. |
| `database.error_type` | Exception class name when the probe failed, else `null`. Never the message. |
| `rag_index.loaded` | Whether a retrieval index is loaded **in this process**. |
| `rag_index.error_type` | Exception class name if the state could not be read, else `null`. |

Both dependency fields are **informational**. They change the field and nothing
else — not `status`, not the HTTP code:

- **A dead database still returns 200 with `status: ok`.** An orchestrator uses
  liveness to decide whether to *restart* the process, and restarting cannot
  repair a database. A probe that failed on a dependency outage would convert
  that outage into a cluster-wide restart loop.
- **The database probe is bounded** (500 ms, on a worker thread). A liveness
  probe that *hangs* is a restart vector too: an orchestrator that times out
  waiting restarts the process just as surely as a 500 would.
- **`rag_index.loaded: false` is normal, not a fault.** It means this process
  has not served a retrieval query yet; a freshly started replica reports
  `false` until its first search or its prewarm completes. Answering the probe
  never loads or builds an index — it reads in-process cache counters — because
  an orchestrator polling every few seconds must not be able to trigger the
  most expensive work the service does.
- **Nothing sensitive is returned.** Connection failures routinely carry the
  DSN, and this route is unauthenticated, so only the exception *class* is
  reported.

`GET /healthz` is an unlisted alias serving the same handler.

### `GET /ready` — readiness

The authoritative, fail-closed signal. It aggregates database, retrieval index,
and (when shared rate limiting is enabled) Redis, and returns **503** when any
required dependency is unavailable, so a load balancer drains the instance
instead of restarting it.

`/health` tells you whether an index is loaded. `/ready` tells you whether
retrieval is good enough to serve traffic. Those are different questions, which
is why `rag_index` on `/health` carries no `meets_deployment_requirement` field.

Passing `/ready` demonstrates bounded engineering dependency availability only.
It is not clinical validation, healthcare compliance, or real-patient
readiness — the response says so in its own `claim_boundary` field.
