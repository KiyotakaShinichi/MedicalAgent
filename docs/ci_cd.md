# CI/CD and Release Gates

This document describes the intended CI/CD flow for the PoC.

## CI steps
- Install Python dependencies
- Run unit tests
- Run agent regression suite
- Run MLE readiness checks
- Generate summary quality evaluation
	- Optional: `python scripts/run_ci_checks.py` to run the full bundle locally

## Release gates
- Safety regression must pass
- MLE readiness hard gates must pass
- Summary quality must be acceptable

## Artifacts
- Data/agent_eval/latest_agent_regression.json
- Data/mle_monitoring/latest_mle_readiness.json
- Data/agent_eval/summary_quality_eval.json

## Deployment notes
- Demo deployments should be supervised and non-clinical
- Production deployment requires security and compliance hardening
