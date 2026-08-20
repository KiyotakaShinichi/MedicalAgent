# Licensing and provenance audit

Status: **incomplete; commercial redistribution is not cleared**.

The repository currently has no root `LICENSE`. Therefore no broad permission
to copy, modify, redistribute, or commercially use the repository's original
source code is granted by this repository. A future code license must cover only
original code and must not be presented as relicensing third-party materials.

## Source code

The Git history currently attributes the repository commits to one author. This
supports authorship traceability but is not itself a license grant. Generated
OpenAPI types and machine-generated reports should retain generator/provenance
metadata where practical.

## Generated artifacts

`Data/evals/`, `reports/`, model metrics, traces, synthetic timelines, and other
evaluation outputs are engineering artifacts. They are not clinical evidence.
Their redistribution status depends on whether they embed third-party source
text, model outputs, or dataset-derived records. Aggregate metrics are generally
lower risk than raw excerpts but still require provenance review.

## Third-party datasets

`Datasets/` contains or references BreastDCEDL/I-SPY, BUSI, and QIN-BREAST-02
materials. Dataset access does not imply unrestricted redistribution. Keep raw
files outside a distributable source package until each dataset's terms,
citation requirements, access conditions, and derived-artifact permissions are
recorded. TCIA, cBioPortal, SEER, GENIE, and other mapped sources retain their
own terms and are not covered by a future NLCare code license.

## Model artifacts and weights

FAISS indexes, fitted estimators, sentence-transformer caches, and future
adapters may inherit model, dataset, or hosting-provider restrictions. The
fine-tuning scaffold records an Apache-2.0 candidate and pinned revision, but no
trained adapter is represented as cleared for clinical or commercial use.

## External knowledge sources

`Data/rag_knowledge_base_chunks.json` includes records marked `CC BY`, records
marked `CC BY-NC-ND`, and many records with `license: null`. `CC BY-NC-ND`
material is not suitable for an unrestricted commercial derivative corpus, and
unknown-license entries require source-by-source review. URLs, titles, source
tiers, allowed-use metadata, retrieval suitability, and license fields must be
preserved. Source-tier approval is a safety policy; it is not a copyright
clearance decision.

## Third-party code and packages

Python and npm packages remain under their upstream licenses. Dependency lock
files establish reproducibility, not relicensing. Any copied snippets must carry
their original attribution and license; no complete snippet inventory currently
proves that this condition is satisfied.

## Distribution boundary

Potentially redistributable after legal review: original source code under a
future explicit license, generated schemas, synthetic-only fixtures, and
aggregate evaluation results that contain no restricted excerpts.

Not presently cleared: raw external datasets, downloaded papers, KB chunks with
unknown or `CC BY-NC-ND` terms, cached model weights, and artifacts containing
substantial source text.

## Required next actions

1. Choose a license for original code after confirming ownership and commercial
   intent.
2. Create a machine-readable source register for every KB document and dataset,
   including license URL, attribution, redistribution, derivatives, and
   commercial-use fields.
3. Quarantine unknown-license and noncommercial content from any distributable
   or commercial build.
4. Add an automated provenance gate that fails distributable builds when any
   included source has unknown or incompatible terms.
