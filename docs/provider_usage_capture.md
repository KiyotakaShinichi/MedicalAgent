# Provider token-usage capture

NLCare separates provider-reported token counts from local character-based
estimates. The readiness check exercises the usage parser and request-scoped
telemetry with a synthetic fixture; it does not call a paid provider.

The controlled probe uses 30 fixed low-risk synthetic education and portal
prompts through `POST /me/chat`. It persists prompt IDs and hashes only, not
prompts or replies. It never spends provider quota automatically.

Readiness only, with no network call:

```bash
python scripts/run_provider_api_path_capture.py
```

Explicit execution requires a provider credential outside source control and
an additional opt-in. For PowerShell:

```powershell
$env:NLCARE_ALLOW_PAID_PROVIDER_PROBE="true"
python scripts/run_provider_api_path_capture.py --execute --request-count 30
```

Non-loopback targets additionally require
`NLCARE_ALLOW_NON_LOOPBACK_PROVIDER_PROBE=true`. After an executed probe,
refresh:

```bash
python scripts/run_cost_latency_report.py
python scripts/run_provider_usage_reconciliation.py
python scripts/run_provider_usage_capture_readiness.py
python scripts/run_ai_trinity_tradeoff.py
```

The target is at least 30 paired requests and at least 80% provider-usage
coverage. Prompt and response content must remain absent from telemetry.
Provider metadata is operational evidence, not audited billing, clinical
validation, or production healthcare readiness.
