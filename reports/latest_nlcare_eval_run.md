# NLCare Evaluation Run

- Status: `needs_attention`
- Generated: `2026-08-11T04:16:58.682745+00:00`
- Commit: `252ae67a2d30935aacd03d0417584524c60b1231`
- KB fingerprint: `93aebd78e618d694`
- Clinical validation: `false`

## Suites

| Suite | Execution | Reported status | Duration ms |
|---|---:|---:|---:|
| integrity | completed | verified_with_external_gaps | 157.61 |
| ai_trinity | completed | needs_attention | 22.338 |
| security_tenant | completed | passed | 574.656 |
| security_poisoning | completed | passed | 10308.159 |
| rag_attribution | completed | acceptable_internal_diagnostic | 859.017 |
| external_feedback | completed | BLOCKED_EXTERNAL | 2.486 |

## Boundary

Reproducible local engineering evaluation over synthetic/internal assets. A passing suite is not clinical validation, independent external evaluation, a production SLO, security certification, or healthcare deployment approval.
