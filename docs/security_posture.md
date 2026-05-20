# Security Posture

This project is a student-built engineering prototype, not a production
healthcare system. The current security posture focuses on reducing obvious
demo and prototype risks.

## Current Controls

- role-scoped patient, clinician, and admin routes
- deterministic prompt-injection and privacy refusal rules
- patient-to-clinician/admin trace access tests
- post-generation validator for blocked medical claims
- dependency scan script when local tools are available
- demo authentication disabled when `ENVIRONMENT=production` unless
  `ALLOW_DEMO_AUTH=true`

## Commands

```bash
python scripts/run_dependency_security_scan.py
pytest tests/test_safety_invariants_property.py -q
```

## Not Claimed

This does not claim HIPAA compliance, production security certification,
penetration testing, SOC 2 readiness, or safe handling of real PHI.
