# Provider token-usage capture

NLCare separates provider-reported token counts from local character-based
estimates. The readiness check exercises the usage parser and request-scoped
telemetry with a synthetic fixture; it does not call a paid provider.

To complete reconciliation, run at least 30 non-patient prompts through the
normal API path with a supported provider credential configured outside source
control. Then refresh:

```bash
python scripts/run_cost_latency_report.py
python scripts/run_provider_usage_reconciliation.py
python scripts/run_provider_usage_capture_readiness.py
```

The target is at least 30 paired requests and at least 80% provider-usage
coverage. Prompt and response content must remain absent from telemetry.
Provider metadata is operational evidence, not audited billing, clinical
validation, or production healthcare readiness.
