# Synthetic Staging Resilience Dossier

This dossier gives reviewers one place to inspect the operational evidence
already produced by executable local drills:

- durable outbox idempotency and lease recovery,
- bounded retry, dead-letter, and audited requeue,
- signing-key rotation, tamper rejection, and replay expiry,
- signed loopback delivery using synthetic recipients,
- temporary SQLite backup and restore,
- deterministic data replay, quarantine, tombstones, and local fallback,
- static Docker Compose validation, and
- managed-vector shadow readiness.

The dossier deliberately keeps unresolved infrastructure work visible:
container runtime recovery is blocked when Docker is unavailable; managed
PostgreSQL point-in-time restore, Azure restore/failover, managed-vector
network execution, real external delivery, and human acknowledgement are not
completed.

Run all source drills first, then:

```powershell
python scripts/run_synthetic_staging_resilience_dossier.py
```

A `strong_local_only_external_blocked` status is not production readiness. It
means the disposable local contracts passed while managed and organizational
evidence remains absent.
