# NLCare buyer technical data room

NLCare is a transferable healthcare-AI **research and engineering platform** for
safety-governed RAG, adversarial agent evaluation, synthetic longitudinal
workflows, evidence provenance, abstention, reproducible MLE, and operational
observability. It is pre-commercial, synthetic-data based, not clinically
validated, not a medical device, and not a production clinical system.

## Start here

1. [Executive summary](EXECUTIVE_SUMMARY.md)
2. [Technical overview](TECHNICAL_OVERVIEW.md)
3. [Assets, IP, licenses, and configuration](ASSETS_IP_AND_CONFIGURATION.md)
4. [Evidence and limitations](EVIDENCE_AND_LIMITATIONS.md)
5. [Deployment and handoff](DEPLOYMENT_AND_HANDOFF.md)
6. [Data, privacy, and security](DATA_PRIVACY_SECURITY.md)
7. [SaaS readiness](SAAS_READINESS.md)
8. [Demo walkthrough](DEMO_WALKTHROUGH.md)
9. [Transfer and diligence checklists](TRANSFER_AND_DILIGENCE.md)
10. [Draft marketplace material](sales/LISTING_DRAFT.md)

## Independent verification

```bash
uv run python scripts/verify_buyer_candidate.py
uv run python scripts/verify_buyer_candidate.py --full
uv run python scripts/build_buyer_package.py --dry-run
```

The verifier resolves the current Git SHA, validates manifests, checks all 757
protected evidence files byte-for-byte, runs existing dependency/offline/secret
contracts, exercises a disposable synthetic demo twice, and validates archive
selection. It does not regenerate scientific evidence or normalize failures.

Machine-readable contracts live in [`config/buyer/`](../../config/buyer/).
This data room is technical diligence material, not legal, regulatory, privacy,
security, medical, investment, or commercial advice.
