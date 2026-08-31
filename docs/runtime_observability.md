# Runtime Observability and Boundary Contracts

NLCare's core runtime observability is vendor-neutral and offline-capable. It
does not require Sentry, Datadog, OpenTelemetry collectors, cloud credentials,
or any outbound network access. A deployment may attach a hosted adapter at its
own data-egress boundary without changing application call sites.

## Structured logging

`backend.logging_config` is the public configuration facade and
`backend.services.structured_logging` is the single implementation. FastAPI
configures it during application lifespan startup. Imports do not configure
global logging.

Application events are JSON and include `timestamp`, `service`, `environment`,
`event`, `level`, `component`, `request_id`, and, when available, `route`,
`method`, and `status_code`. Framework logs remain on Python's standard
`logging` pipeline. The dynamic stdout handler resolves the current process
stream for each event, preventing stale or closed capture streams.

Redaction is applied before emission. Authorization values, cookies, session
tokens, API keys, webhook secrets, patient identifiers, prompts, messages,
symptoms, findings, impressions, notes, common identifiers, and credential-like
string patterns are removed. Full request and response bodies are not logged.

## Request correlation

`X-Request-ID` is the canonical HTTP correlation header. A caller-supplied ID
is preserved only when it is 1-128 characters and matches the bounded grammar
`[A-Za-z0-9][A-Za-z0-9._:-]*`. Missing, malformed, whitespace-bearing, or
oversized values are replaced by an opaque server-generated ID.

The same ID is stored in request state and a `ContextVar`, returned in the
response header, and inherited by service-layer structured events. It is a
correlation value, never a patient identifier or metrics label.

## Error reporting

`backend.services.error_reporting.ErrorReporter` is the adapter contract.
`StructuredLogErrorReporter` is the default and emits only error category,
exception class, bounded route/method context, and request ID. It never emits
the exception message, stack trace, or request payload. `NoOpErrorReporter` is
available for constrained embeddings.

Reporter calls are failure-isolated. If an installed adapter raises, NLCare
still returns its bounded, correlated error response. Expected FastAPI HTTP and
Pydantic validation errors retain their existing status semantics and are not
reported as catastrophic runtime exceptions.

## Health and readiness

`GET /health` and its `/healthz` alias are liveness probes. They return 200 when
the process is alive. Database and loaded-index fields are informational and do
not cause restart loops.

`GET /ready` and `/readyz` are traffic-readiness probes. Database and local RAG
runtime checks are required. Redis is required only when shared rate limiting
is explicitly enabled. Optional LLM/vector providers are not contacted and do
not make the process unready when the local serving contract remains valid.
Required dependency or configuration failures return 503. Probe details contain
only bounded status fields and exception class names, never URLs, paths, or
credentials. Runtime readiness remains an engineering signal and does not imply
clinical validation or healthcare production readiness.

## Metrics

`backend.services.runtime_metrics` provides a lightweight process-local sink
for request count, HTTP error count, aggregate request latency, and readiness
outcomes. Labels are restricted to HTTP method, route template, and status
family. Request IDs, patient IDs, raw URLs, messages, and medical content are
not labels. No public metrics endpoint is exposed in this track; a future
Prometheus or OpenTelemetry adapter can implement the same sink contract.

Safety actions and RAG decisions remain observable through their existing
typed trace and evaluation artifacts. This track does not change those
decisions, duplicate patient content into logs, or regenerate evidence.

## Validation and test hygiene

`backend.services.api_boundary_inventory` classifies every mutating operation
as a typed request body, query/path-only action, multipart upload, or justified
raw-body exception. The only raw-body exception is the signed n8n delivery
receipt, whose HMAC must cover the exact bytes before parsing.

Patient symptom, CBC, and chat boundaries reject unknown fields; CBC values
also reject NaN and infinity. Existing domain validators continue to protect
agent-created records that do not pass through HTTP schemas.

Generators keep production output defaults, but tests inject temporary output
roots and manifest paths. Python bytecode and runtime cache files are ignored
and must never be tracked. All observability tests run with
`NLCARE_TEST_OFFLINE=true` and require no monitoring credentials or network.
