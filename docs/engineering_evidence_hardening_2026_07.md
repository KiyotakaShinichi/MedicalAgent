# Engineering evidence hardening, July 2026

This pass closes several internal engineering-evidence gaps without changing
NLCare's clinical boundary. It adds no diagnosis, treatment, prognosis,
medication, genetics, or tumor-marker authority.

## Agent execution policy

`backend/services/agent_execution_policy.py` adds a typed state path and
fail-closed execution envelope. It limits tool calls, workflow steps, and write
actions; requires an explicit write-confirmation contract; blocks unknown or
medical-authority tools; and treats memory as non-authoritative unless its
provenance is trusted. The offline policy artifact contains six case-level
allow/block checks and performs no live patient write.

Run:

```powershell
python scripts/run_agent_execution_policy_eval.py
```

## RAG degradation drills

The local drill uses disposable indexes and forces the supported sparse
fallback so it never downloads an encoder. It checks corrupt-index recovery,
knowledge-fingerprint refresh, dense-unavailable behavior, missing optional
metadata, and empty queries. It does not test managed vector recovery,
generation quality, or a production outage.

```powershell
python scripts/run_rag_degradation_resilience_eval.py
```

## XAI retraining sensitivity

The stricter audit refits the logistic engineering baseline across 12
patient-level split seeds and compares global and local grouped explanation
rankings. It is deliberately reported as `needs_attention` because exact
global ordering is unstable. This negative result is retained rather than
smoothed or tuned away.

```powershell
python scripts/run_xai_retraining_stability_audit.py
python scripts/run_xai_fidelity_audit.py
```

## Software supply chain

The supply-chain artifact derives CycloneDX 1.5 SBOMs from the locked Python
and frontend dependency graphs, records lock fingerprints, and runs the
repository secret scanner without storing matched secret values. Vulnerability
results remain in the separate dependency-security artifact. Container
scanning is explicitly incomplete.

```powershell
python scripts/run_software_supply_chain_evidence.py
python scripts/run_dependency_security_scan.py
```

## Synthetic automation staging

`docker-compose.synthetic-automation.yml` defines loopback-only n8n and
MailHog services. The inactive workflow verifies a canonical HMAC signature,
uses constant-time comparison, enforces a five-minute freshness window,
rejects blocked payload fields, and addresses only a synthetic `.invalid`
recipient. SMTP credentials must be configured manually in n8n.

```powershell
python scripts/run_synthetic_automation_staging_readiness.py
docker compose -f docker-compose.synthetic-automation.yml config --quiet
```

Static and compose validation do not mean the workflow was imported or that an
email was delivered. Real email, SMS, Viber, emergency coverage, and clinician
acknowledgement remain disabled and unproven.

## Claim boundary

All artifacts in this pass are synthetic/internal software evidence. They do
not establish clinical validation, clinician approval, real patient safety,
patient benefit, regulatory compliance, managed-cloud reliability, security
certification, or production healthcare readiness.
