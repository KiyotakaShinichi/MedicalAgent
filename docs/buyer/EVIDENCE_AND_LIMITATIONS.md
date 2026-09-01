# Evidence index and limitations

## Evidence families

| Family | Purpose and authoritative location | Positive evidence | Negative evidence / interpretation |
|---|---|---|---|
| DEP-001 | `Data/evals/safety/dep001*/` and deployment reports | Layered fail-closed architecture and extensive internal regressions | Official independent behavioral failures remain `BLOCKED_BEHAVIORAL`; do not rerun consumed banks |
| RAG | `Data/evals/rag/` | Source-tier correctness, traceability, baseline/oracle diagnostics | Full governed stack has not proven raw recall superiority; pruner regressed precision |
| Safety/adversarial | `Data/evals/safety/` | Deterministic guardrails, multilingual and fault-path suites | Internal/tuning-used success is not independent validation; held-out weaknesses remain visible |
| ML/MLE | `Data/evals/models/` | Leakage, temporal CV, calibration, uncertainty, lineage, promotion gates | Synthetic saturation, target mismatch, and shortcut risks prevent clinical promotion |
| XAI | model explanation artifacts and admin surfaces | Row-level traces, feature/explanation evidence, missingness/abstention context | Explanations describe simulator/model behavior, not medical causality |
| Provenance | registries, manifests, contamination map, source governance | Artifact lineage and evidence maturity are explicit | Several external/no-read reviews remain prepared but incomplete |
| Fresh clone / CI | `.github/workflows/`, `scripts/check_fresh_clone_offline.py` | Hermetic offline verification and green R4 CI baseline | Dependency/model provisioning is a separate declared step |
| Runtime operations | `Data/evals/ops/`, `/health`, `/ready` | Structured logs, request IDs, redaction, runtime boundaries | Metrics are process-local; no durable exporter or vendor error adapter |

All 757 protected files are enumerated in
`config/buyer/protected_evidence_manifest.json` with SHA-256 digests. SP1 indexes
them; it does not refresh or rewrite them.

## Required disclosures

- Primary data and ML training are synthetic. Synthetic quality proxies are not
  clinical realism or external validity.
- No prospective validation, real patient data, IRB/ethics approval, clinician
  sign-off, regulatory clearance, medical-device status, or production clinical
  deployment exists.
- NLCare must not diagnose, prognose, recommend treatment, change dose, interpret
  genetic risk/VUS or tumor markers as conclusions, or replace a care team.
- DEP-001 official external evidence contains genuine safety failures. A fresh,
  independently authored no-read evaluation is the only legitimate replacement.
- The project has a single-human-author history and no represented customers,
  users, revenue, hospital partners, or production outcomes.
- RAG evidence remains mostly internal/in-sample; held-out v2 and adjudication
  workflows are prepared but not completed.
- Current ML and XAI evidence is simulator-built and monitor/review/context only.
- Demo identity, local SQLite, and `.env` configuration are not production health
  data controls.
- Tenant identifiers and authorization seams exist, but complete hostile
  multi-tenant isolation, privacy lifecycle, deletion/export, immutable audit,
  and compliance assurance are not established.
- Some legacy services remain oversized; this is maintainability debt, not hidden.
- Optional vendor services and downloaded models/papers are not transferred.

The release/evidence gates are engineering controls. A passing gate must never be
described as clinical readiness or a real-world safety guarantee.
