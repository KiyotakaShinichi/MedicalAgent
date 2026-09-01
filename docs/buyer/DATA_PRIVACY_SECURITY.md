# Data, privacy, and security inventory

NLCare is approved here only for synthetic engineering data. Production use with
real health data would require independent legal, privacy, security, clinical,
and regulatory programs. No HIPAA, GDPR, Philippines Data Privacy Act, or other
compliance claim is made.

| Data class | Location | Patient-like? | Current retention/deletion | Transfer status | Hosted-production requirement |
|---|---|---:|---|---|---|
| Synthetic journeys/training | `Data/complete_synthetic_*` | Yes, synthetic | Tracked/versioned; delete through source control policy | Included, rights review | Data governance, minimization, lineage review |
| Buyer demo SQLite | `Data/test_tmp/buyer_demo/` | Yes, synthetic | Disposable; marked reset command deletes only demo files | Generated, not archived | Managed DB, encryption, retention, backup/restore |
| Runtime DB | `DATABASE_URL` | Potentially | Operator-managed; no complete data-subject lifecycle | Not transferred as live data | Formal retention, export/deletion, access controls |
| Uploads | `Data/uploads/` | Potentially | Gitignored; disabled in production-shaped profile | Not included | Malware scan, object policy, encryption, deletion |
| Logs/traces | stdout/runtime sink and DB logs | May contain synthetic prompts/metadata | Redacted structured logging; deployment policy absent | Runtime only | Central retention, access, deletion, tamper evidence |
| Caches/indexes | `Data/rag_index`, Redis, local caches | KB/query-derived | Rebuildable/TTL-dependent | Generated | Namespace isolation, eviction, encryption, purge |
| Evaluation artifacts | `Data/evals/` | Synthetic/test prompts | Tracked and immutable by evidence policy | Included | Preserve lineage and classification |
| External dataset bridges | `Data/external_*`, `Data/breastdcedl_*` | Public/research records | Tracked research snapshots | Excluded from buyer archive | Reacquire under terms; assess privacy/licensing |
| Reports/screenshots | `reports/`, generated outputs | Potentially | Only intentional tracked evidence should persist | Case-specific | Redaction and review before export |
| Research papers | gitignored optional corpus | No patient records expected | Operator-acquired | Not included | Source terms, secure storage, provenance |

## Security/account boundary

The transfer excludes all keys, OAuth/client secrets, LLM/vector/cloud accounts,
personal email, DNS/TLS, SSH keys, cookies, browser profiles, and local storage.
The archive uses tracked files only, rejects secret-like filenames, excludes
runtime databases/logs/caches, and invokes the repository secret scanner.

Demo auth is intentionally weak and synthetic. It is disabled in staging and
production unless explicitly overridden. OIDC and tenant-scoped infrastructure
keys are implementation seams, not certification of identity or tenant
isolation. Before hosting, a buyer must threat-model every route, validate
authorization against a real IdP, establish rate limits and session controls,
perform independent penetration/privacy reviews, and implement incident and
deletion workflows.

Historical Git author email remains in commit metadata and is not rewritten.
Current-tree personal CV exports and runtime logs were removed from this
candidate. Three historical consumed-holdout compatibility paths remain in
protected/safety code or evidence and are documented rather than modified.
