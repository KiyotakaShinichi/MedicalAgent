# Production Readiness Boundary

NLCare is now more production-shaped as an engineering project, but it is **not**
healthcare-production-ready.

Run:

```bash
python scripts/run_production_readiness_boundary.py
```

Output:

```text
Data/evals/governance/latest_production_readiness_boundary.json
```

The artifact separates:

- engineering readiness signals: release gates, eval artifacts, traceability,
  latency observability, external-review packets
- healthcare-production blockers: no external review, no clinician-reviewed
  labels, no clinical validation, no IRB/ethics approval, no PHI/compliance
  review, no real-world monitoring or deployment SLO

Allowed wording:

> Production-shaped engineering prototype with release gates, eval artifacts,
> traceability, and explicit clinical boundaries.

Blocked wording:

- production healthcare system
- clinically validated AI
- safe for real patient care
- clinician-approved
- HIPAA/PHI-compliant deployment
- treatment recommendation system
