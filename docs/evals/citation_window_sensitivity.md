# Citation Window Sensitivity

This is internal engineering evidence only. It is not clinical validation, not a patient-safety claim, and not proof that the RAG stack is clinically grounded.

- Status: `acceptable`
- Configuration: `hybrid_rrf_query_rewrite_parent_child_source_tier`
- Case count: `74`
- Recommended cited-context K: `1`
- Promotion recommendation: `do_not_promote_without_more_evidence`

| cited_context_k | citation_precision | delta_vs_k5 | cited_window_support_rate | low_precision_cases |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6081 | 0.0865 | 0.6081 | 29 |
| 2 | 0.5811 | 0.0595 | 0.7162 | 21 |
| 3 | 0.5721 | 0.0505 | 0.7703 | 30 |
| 5 | 0.5216 | 0.0 | 0.8243 | 28 |

## Interpretation

Citation precision can improve when fewer chunks are cited, but this does not prove retrieval improvement. Any live change should be A/B tested against generated-answer claim support and refusal behavior.

Do not present this as clinical validation or as proof that retrieval is solved.
