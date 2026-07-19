# RAG governance trade-off

NLCare keeps one canonical report that places raw retrieval effectiveness,
grounding proxies, source governance, and latency side by side. This prevents
the source-governed stack from being described as a general retrieval
improvement when the internal baseline does not support that claim.

On the current frozen internal goldset, BM25 has higher Recall@10 than the full
source-governed stack. The full stack has substantially higher source-tier
correctness, but also greater latency. It is retained for allowed-use and
source-governance behavior, not because raw retrieval superiority is proven.

The external no-read holdout remains incomplete. These measurements are
internal engineering evidence only and are not clinical validation, external
generalisation, healthcare production readiness, or proof of patient benefit.
