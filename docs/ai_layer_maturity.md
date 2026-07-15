# AI / RAG Layer Maturity

NLCare's AI layer is designed as a safety-first orchestration system, not
an open-ended medical chatbot.

## Current Controllable Guarantees

- Input safety routing runs before retrieval or generation.
- The patient agent separates casual support, portal help, education RAG,
  urgent safety routing, security/privacy refusals, and treatment-boundary
  refusals.
- Retrieval is source-governed through tier, allowed-use, and staleness
  metadata.
- Patient-facing RAG answers are checked with claim-level citation validation.
- A post-generation validator blocks diagnosis, treatment, prognosis, dosage,
  genetic-risk, and tumor-marker overclaims.
- Refusal and insufficient-evidence outcomes are treated as successful behavior
  when the request is unsafe or under-supported.
- RAG calls persist replayable traces with intent, rewrite, sources, source
  tiers, claim validation, post-generation validation, and final telemetry.
- Taglish/code-switched safety parity is evaluated as a first-class artifact.

## Proof Commands

```powershell
python scripts\run_rag_claim_validation_eval.py
python scripts\run_nli_claim_validation_eval.py
python scripts\run_live_rag_eval.py
python scripts\run_rag_tier_ablation.py
python scripts\run_release_gate.py
```

## Key Artifacts

- `Data/evals/rag/latest_claim_level_citation_eval.json`
- `Data/evals/rag/latest_nli_claim_validation_eval.json`
- `Data/evals/rag/latest_live_rag_eval.json`
- `Data/evals/rag/latest_rag_tier_ablation.json`
- `Data/evals/rag/latest_kb_source_governance.json`
- `Data/evals/safety/latest_taglish_safety_parity.json`

## Remaining Limits

The claim validator has an optional NLI path, but CI still uses lightweight
heuristics with explicit contradiction checks. NLI defaults to local-cache-only
loading so the release gate does not download a large model unexpectedly.
To deliberately run the NLI path, install `transformers` and `torch`, then set:

```powershell
$env:ONCOTRACK_RAG_CLAIM_VALIDATOR="nli"
$env:ONCOTRACK_NLI_ALLOW_DOWNLOAD="1"
python scripts\run_nli_claim_validation_eval.py
```

The high-risk contradiction heuristic remains a safety veto over NLI because
small generic MNLI models can over-entail medical inversions such as
"St. John's wort is safe with chemotherapy" when the source actually says to
ask the oncology team or pharmacist. This is engineering safety evidence, not
clinical correctness evidence.
