# Azure Non-Patient Shadow Runbook

## Boundary

This runbook is for a disposable engineering development resource group and
curated non-patient knowledge only. It does not authorize patient data, prove
clinical validation, certify security/compliance, or make NLCare production
healthcare ready.

The local FAISS/BM25 route remains canonical. Azure AI Search is a shadow
candidate only.

## Prerequisites

1. Use a separate development subscription with a reviewed spending limit.
2. Install Azure CLI and authenticate with an engineering identity.
3. Confirm the target subscription:

```powershell
az account show --output table
```

4. Compile without contacting Azure:

```powershell
.\tools\bin\bicep.exe build infra\azure\main.bicep --stdout > $null
python scripts/run_cloud_infrastructure_readiness.py
```

5. Choose an engineering email address for optional alerts. Never use a
patient address.

## What-if First

Create a disposable resource group only after a human reviews the subscription
and region:

```powershell
az group create --name nlcare-shadow-dev-rg --location southeastasia
az deployment group what-if `
  --resource-group nlcare-shadow-dev-rg `
  --template-file infra/azure/main.bicep `
  --parameters prefix=nlcare environment=dev location=southeastasia `
  --parameters deployPrivateNetworking=true deployManagedSearch=true `
  --parameters deployComputeEnvironment=false deployMessaging=false deployPostgres=false `
  --parameters deployOperationalAlerts=false deployCostControls=false `
  --parameters allowPublicNetworkAccess=false
```

Stop if `what-if` shows resources outside the disposable group, a public
endpoint, an unreviewed role assignment, or an unexpected paid service.

## Deployment Decision

Deployment is deliberately not automated from a developer laptop. A private
Search service cannot be queried from the public internet; index provisioning
and evaluation must run from an approved workload or runner attached to the
VNet. This prevents "temporary" public exposure from becoming the default.

After a reviewed deployment, acquire an Entra bearer token from that workload,
set the shadow variables without committing them, and validate the index:

```powershell
$env:NLCARE_VECTOR_BACKEND='azure_ai_search'
$env:NLCARE_MANAGED_VECTOR_SHADOW_ENABLED='true'
$env:NLCARE_MANAGED_VECTOR_ALLOW_NETWORK='true'
$env:AZURE_SEARCH_ENDPOINT='https://<service>.search.windows.net'
$env:AZURE_SEARCH_INDEX_NAME='nlcare-kb-shadow-v1'
$env:AZURE_SEARCH_BEARER_TOKEN='<short-lived-token>'
python scripts/run_azure_search_index_readiness.py --apply
python scripts/run_managed_vector_shadow_sync.py --apply --batch-size 50
```

Only curated gold records from
`Data/lakehouse/gold/vector_records.jsonl` may be loaded. The adapter rejects
unapproved namespaces, unknown metadata, patient identity fields, and any
record not marked `data_scope=curated_non_patient_kb`.

## Frozen Shadow Comparison

Run:

```powershell
python scripts/run_managed_vector_shadow_comparison.py
```

Review these jointly:

- Recall@5 and Recall@10;
- MRR and NDCG@10;
- citation precision and claim-support rate;
- unsupported-context rate;
- refusal and source-tier correctness;
- p50/p95 query latency;
- measured Azure cost over the observation window;
- ingestion freshness;
- delete propagation;
- managed outage fallback.

Do not promote Azure AI Search because one retrieval metric is higher.
`quality_governance_joint_improvement_proven` must be true and the operational
evidence fields must be complete. Even then, the result is internal engineering
evidence, not clinical validation.

## Recovery And Deletion

Before any shadow promotion:

1. Delete a synthetic test chunk from gold and verify removal from Search.
2. Rebuild the index from the content-addressed manifest.
3. Block Search access and verify the local route remains available.
4. Restore a disposable PostgreSQL server to a new name if PostgreSQL is
enabled.
5. Record recovery-point and recovery-time observations.

The local reliability artifact does not substitute for these live drills.

## Teardown

After exporting redacted engineering metrics:

```powershell
az group delete --name nlcare-shadow-dev-rg --yes --no-wait
```

Verify the resource group is gone and no detached resources or budgets remain.
