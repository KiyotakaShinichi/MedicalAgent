# Deployment boundary

> **`production_shaped_not_healthcare_production_ready`** is the
> permanent status label of this project. The
> `deployment_boundary_check` artifact's `status_label` field is
> test-locked to that exact string. A future contributor cannot
> ratchet it past that without explicitly editing
> `backend/services/deployment_boundary_check.FIXED_STATUS_LABEL` and
> breaking the test suite — which is the intended forcing function.

## What this project IS, deployment-wise

- A FastAPI backend + React frontend that can be started with
  `docker-compose up`.
- A 127-artifact release gate that passes against engineering-readiness
  criteria.
- A documented startup path, a `.env.example`, a `ship.py` local-smoke
  command, health-check documentation, and a security-controls doc.

## What this project is NOT, deployment-wise

- **Not HIPAA compliant.** No business associate agreement, no PHI
  handling assessment, no audit logging that meets HIPAA's audit
  controls.
- **Not SOC 2 / HITRUST.** No control attestation, no continuous
  monitoring vendor, no formal risk register.
- **Not FDA / CE / regulatory cleared.** No SaMD classification, no
  pre-market submission, no IFU.
- **Not clinically deployed.** No clinical site, no clinical workflow
  integration, no clinician sign-off.
- **Not patient-safe under real use.** No real-patient evaluation
  exists.

The `claim_boundary` field in the artifact says all of this
verbatim, and the test suite asserts:

- `status_label == "production_shaped_not_healthcare_production_ready"`
- `clinical_validation == false`
- `no_hipaa_compliance_claim == true`
- `no_clinical_deployment_claim == true`
- `what_this_does_not_certify` includes HIPAA, SOC 2, FDA, clinical
  safety, real-world patient privacy, and production healthcare
  deployment.

## What the check verifies (and what it doesn't)

| Check | What passing means | What passing does NOT mean |
|---|---|---|
| `env_example_present` | A `.env.example` exists so a new operator can boot the system. | The example contains the right values for any specific environment. |
| `docker_compose_present` | A `docker-compose.yml` is wired. | The compose file is production-grade. |
| `deployment_readiness_doc_present` | A deployment-readiness doc exists. | The doc is sufficient for healthcare deployment. |
| `deployment_boundary_doc_present` | This doc exists. | The boundary is a substitute for real compliance. |
| `health_check_doc_present` | A health-check / monitoring doc exists. | The agent has a production-grade liveness/readiness contract. |
| `local_smoke_command_present` | A `scripts/ship.py` smoke command exists. | The smoke covers production failure modes. |
| `security_controls_doc_present` | A `docs/security_controls.md` exists. | Controls have been independently audited. |
| `demo_credentials_gated_off_in_production` | Demo auth is OFF when `ENVIRONMENT=production` unless explicitly flagged. | Demo credentials are safe to ship; they are not. |

## What we never claim

- Healthcare production readiness.
- HIPAA / SOC 2 / HITRUST compliance.
- Clinical deployment readiness.
- Patient safety under real use.
- Real-world data privacy.
- Regulatory clearance.

The `deployment_boundary_check.json` artifact's
`what_this_does_not_certify` list is the canonical version of those
non-claims.

## Related

- [`docs/ten_out_of_ten_under_constraints.md`](ten_out_of_ten_under_constraints.md)
- [`docs/evals/eval_contamination_harmonization.md`](evals/eval_contamination_harmonization.md)
- [`docs/portfolio_safe_wording.md`](portfolio_safe_wording.md)
