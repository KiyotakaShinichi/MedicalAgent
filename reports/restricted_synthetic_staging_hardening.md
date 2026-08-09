# Restricted Synthetic Staging Hardening

## Implemented Controls

- Hash-only demo session persistence; raw bearer tokens are returned once.
- Browser bearer storage moved from persistent local storage to session storage.
- Strict profiles require browser OIDC authorization code plus PKCE settings.
- Synthetic-only mutating API boundary with explicit data-classification header
  and synthetic patient namespaces.
- Uploads disabled by default in strict profiles; quarantine, strict decoding,
  content/type matching, and an external fail-closed scanner are required when
  enabled.
- High-risk RAG claims require semantic entailment in strict profiles; unavailable
  semantic validation becomes abstention rather than lexical acceptance.
- Leased automation worker wired into Compose.
- Multi-stage, non-root distroless Python backend with a shell-free entrypoint.
- Runtime import probe prevents a shallow health endpoint from masking missing
  serving dependencies.

## Executed Evidence

The disposable loopback stack has seven running services and passes backend,
frontend, n8n, MailHog, Postgres, Redis, and backend dependency-import probes.
The drill completed a Postgres restore, imported an inactive n8n workflow, and
received a synthetic-only message in MailHog. It performed no real external
delivery and processed no patient data.

Focused hardening and evidence tests pass. The required staging assurance passes
25 tests with clean Python/npm dependency evidence. The release gate passes with
the known adversarial-generalization warning preserved.

## Stop-The-Line Result

The first exact-image Trivy scan found 2 critical and 40 high findings, including
18 high/critical findings with published fixes. The runtime was then moved to
Python 3.13 on Debian 13 distroless and rebuilt end to end. The fresh exact-image
scan contains 0 critical and 15 high findings, with no published fix currently
reported for those 15. `latest_container_security_scan.json` therefore still
records `BLOCK_PUBLIC_DEPLOYMENT` and appears as a release warning. This result
is not waived by the otherwise passing engineering gate.

## Honest Boundary

This supports a local, loopback-only, synthetic staging demonstration. It does
not establish a secure public deployment, live identity-provider integration,
real external alert delivery, privacy compliance, clinical validation, patient
benefit, or production healthcare readiness.
