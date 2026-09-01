# Assets, IP, licenses, and configuration

## Authoritative inventories

- [Asset manifest](../../config/buyer/asset_manifest.json)
- [Technical license inventory](../../config/buyer/license_inventory.json)
- [Configuration matrix](../../config/buyer/configuration_matrix.json)
- [Buyer candidate manifest](../../config/buyer/candidate.json)
- [Package policy](../../config/buyer/package_policy.json)

These are machine-validated by `python scripts/verify_buyer_candidate.py`.

## First-party boundary

The repository has no root `LICENSE`. The history shows one human author plus
automated dependency commits, but this data room does not infer title or draft
binding transfer terms. A buyer should obtain signed representations covering
source, documentation, synthetic generators/data, model artifacts, and any
contractor/contributor rights, then choose licensing or assignment terms with
counsel.

## Third-party boundary

Acquiring NLCare source does not transfer ownership of packages, models,
datasets, publications, icons, provider services, or cloud accounts. Python and
frontend CycloneDX inventories already exist under `Data/evals/ops/sbom/`.
Dense/cross-encoder weights are downloaded separately under their model terms.
Optional paper full text is gitignored. BreastDCEDL is noncommercial; its source
data and derived pathways require separate review. Controlled/public datasets
must be reacquired under current source terms.

The deterministic archive excludes external bridge/source files and
BreastDCEDL-derived model paths while preserving every protected evaluation
artifact. This is a packaging control, not a legal conclusion.

## Configuration contract

`.env.example` is the canonical, secret-free operator template. The generated
matrix contains every declared variable and fails verification if it drifts.
Categories cover APP, AUTH, DATABASE, RAG, LLM, VECTOR, ML, OBSERVABILITY,
AUTOMATION, DEMO, and INFRA. Secret fields carry placeholders only.

Never transfer `.env`, API keys, OAuth secrets, private URLs, cookies, SSH keys,
browser profiles, cloud configuration, personal email access, or local storage.
The buyer provisions all external accounts and rotates every credential.

## Diligence gaps

| Area | Current state | Transferability | Buyer risk/action | Priority |
|---|---|---|---|---|
| First-party source rights | No root license/assignment in repo | Review required | Execute ownership and transfer/license documents | Critical |
| Curated KB summaries | Tracked source-backed summaries; no formal rights attestation | Review required | Review authorship, attribution, and commercial reuse | High |
| External data | Mappings/derived research files under varied terms | Package excludes risky source paths | Reacquire and review per dataset | High |
| Models | Synthetic and external-derived artifacts coexist | Lineage-specific | Exclude/retrain anything without commercial lineage | High |
| Providers | Optional seams, no accounts transferred | Buyer-provisioned | Establish accounts, agreements, budgets, regions | Medium |
| Frontend assets | Lucide dependency plus starter/custom assets | Review required | Confirm origin or replace before commercial use | Medium |

This is technical diligence, not legal advice.
