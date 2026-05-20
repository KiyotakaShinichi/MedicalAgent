# RAG Regression Debug Runbook

1. Open the failing case in the live RAG, claim citation, or retrieval ablation
   artifact.
2. Classify the failure:
   - intent routing
   - retrieval miss
   - source-tier filtering
   - citation assembly
   - semantic contradiction
   - post-generation validator
   - cache/citation restoration
3. Reproduce with the smallest script possible.
4. Check trace replay for query rewrite, retrieved sources, source tiers, claim
   validation, post-generation decision, and final answer.
5. Patch general behavior, not exact case text.
6. Rerun live RAG, claim citation, over-refusal, and safety banks.

Keep unsafe answer rate at zero. Do not improve pass rate by weakening refusal.
