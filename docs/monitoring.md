# Monitoring and Observability

This document outlines the monitoring posture for the PoC.

## Key telemetry
- App event logs (errors, patient input, model training actions)
- Prediction audit logs
- Agent RAG evaluation logs (grounding, hallucination, guardrails)
- Cache hit and expiration metrics

## Metrics to watch
- Guardrail failure rate
- Safety boundary refusal accuracy
- Cache policy accuracy
- Latency and token cost proxies
- Drift proxy and data coverage

## Alert ideas
- Guardrail failures > 0
- Missing citations for knowledge answers
- Spike in unsafe advice or boundary violations
- Data coverage or drift status in "failed"

## Traceability
- Request IDs attached to app event logs and RAG evaluation logs
- Admin analytics and evaluation reports stored under Data/
