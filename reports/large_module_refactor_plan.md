# Large-module responsibility review

Assessment date: 2026-08-20

Generated schemas, data artifacts, and one-file report generators were excluded
from the initial refactor queue. File length alone is not a reason to split a
module.

| Module | Approx. LOC | Mixed responsibilities | Decision |
|---|---:|---|---|
| `backend/api/routers/admin_eval.py` | 1,409 | Route registration, trace serialization, artifact loading, evaluation execution, and system-health endpoints | Split the observability/trace routes into one sub-router; preserve URLs and dependencies |
| `backend/services/mle_readiness.py` | 1,251 | Readiness orchestration, contract gates, lifecycle checks, and statistical diagnostics | Move the self-contained hybrid-weight and temporal-generalization diagnostics into a statistics module |
| `backend/services/support_chat_agent.py` | 1,213 | Turn orchestration plus extraction, tool planning, and pending-write state | Defer; high safety blast radius and ongoing DEP-001 behavior evidence make this a poor opportunistic refactor |
| `backend/services/rag_evidence_envelope.py` | 1,126 | Envelope construction, release authorization, parsing, abstention, and telemetry | Defer; this is a safety-critical release boundary and should move only with dedicated equivalence/fault tests |
| `frontend-react/src/pages/admin/sections/SafetyCenterSection.tsx` | 1,075 | Data fetching, summary layout, and many benchmark panels | Defer to a UI-specific pass with screenshot and accessibility regression coverage |

## Selected split 1: admin observability routes

Extract agent trace-list, RAG trace-replay, and system-health endpoints into
`backend/api/routers/admin_eval_observability.py`. The parent factory includes
the sub-router. This creates an ownership boundary around operational views and
removes trace serialization from the evaluation registry without changing
public routes.

Acceptance: route paths and RBAC dependencies remain identical; existing access
control and trace tests pass; OpenAPI type drift remains clean.

Implemented: `admin_eval.py` decreased from 1,409 to 1,252 lines. The extracted
router is covered by `tests/test_admin_eval_observability.py`, while
`tests/test_rag_trace_replay.py` exercises the persisted trace response and
admin-only access contract.

## Selected split 2: MLE statistical diagnostics

Extract hybrid-weight ablation and synthetic temporal-generalization diagnostics
into `backend/services/mle_readiness_statistics.py`. The parent readiness report
imports the functions under their existing private names so output contracts are
unchanged.

Acceptance: existing MLE readiness tests and report schema pass unchanged; no
model, threshold, dataset, or metric definition changes.

Implemented: `mle_readiness.py` decreased from 1,251 to 1,061 lines. Existing
private imports remain available through aliases. The direct and re-exported
contracts are compared in `tests/test_mle_readiness_statistics.py`.

## Deferred debt

The next refactor should target `support_chat_agent.py`, but only after a fresh
development bank and a new independent holdout protocol exist. The current
consumed DEP-001 evidence must never be used as a refactor tuning target.
