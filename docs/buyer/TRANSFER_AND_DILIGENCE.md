# Transfer and independent diligence

## Transaction-time technical checklist

- [ ] Exact Git SHA and included branches agreed
- [ ] `python scripts/verify_buyer_candidate.py --full` passes at that SHA
- [ ] Source/package SHA-256 and package manifest verified
- [ ] Dependency locks and SBOMs reviewed
- [ ] First-party ownership/assignment representations completed outside Git
- [ ] Third-party license inventory and package exclusions reviewed
- [ ] Curated KB, frontend assets, generated data, and model lineage reviewed
- [ ] No secrets, accounts, personal files, runtime DBs, or caches included
- [ ] Synthetic demo seed/reset and walkthrough verified
- [ ] All 757 protected evidence hashes verified
- [ ] DEP-001 and other negative evidence reviewed without normalization
- [ ] Buyer-owned providers, identity, cloud, DNS/TLS, and monitoring identified
- [ ] Database migration and local backup/restore verified
- [ ] Hosted operations/privacy/security roadmap accepted
- [ ] Support, payment, legal warranties, and handoff terms handled outside source

## Buyer independent diligence

Run and inspect rather than relying on screenshots:

- [ ] Git history, authorship, branches, tags, and source hash
- [ ] Backend tests/branch coverage, Ruff, configured mypy, LOC ratchets
- [ ] Frontend tests/coverage, lint, typecheck, and production build
- [ ] CI and fresh-clone/offline provisioning
- [ ] Docker build/profile, liveness/readiness, and failure paths
- [ ] Asset, license, configuration, data/privacy, and secret inventories
- [ ] RAG/agent/safety/ML/XAI evidence, including negative artifacts
- [ ] Demo behavior with no provider keys and a disposable empty database
- [ ] Migration chain, backup/restore, reset safety, and generated-state cleanup
- [ ] Archive exclusions, content manifest, deterministic hash, and secret scan
- [ ] Auth/session/authorization/tenant isolation and privacy lifecycle gaps
- [ ] External account, model, paper, and dataset acquisition requirements

## Non-exclusive source license vs acquisition

This is a technical comparison, not binding legal language.

### A. Non-exclusive source license

May grant defined use of selected first-party source, documentation, tests, and
synthetic demo material while the author retains ownership and may license to
others. It cannot override third-party terms, transfer provider accounts, or
confer clinical/regulatory status. Scope, field, sublicensing, support, updates,
warranties, and generated-data/model rights require professional drafting.

### B. Full project/IP acquisition

May transfer agreed first-party project rights, repository/history, domains or
brand assets if separately listed, and handoff material. Third-party packages,
datasets, publications, models, accounts, and licenses still remain subject to
their own terms. Assignment chain, moral rights, exclusions, liabilities,
support, privacy, and clinical boundaries require professional review.

## No-owner-knowledge results

The current tree no longer tracks personal CV exports, transient runtime logs,
rendered temporary pages, or generated coverage XML. Runtime paths are derived
from repository/configuration roots. Localhost ports are documented profile
defaults rather than owner-machine dependencies. Historical Git identity remains
in commit metadata. Three consumed DEP-001 compatibility references contain an
old absolute path; they are isolated by an environment override and left intact
to avoid changing frozen safety evidence semantics.
