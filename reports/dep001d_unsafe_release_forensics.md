# DEP-001D Unsafe-Release Forensics

## Evidence boundary

This postmortem examines only the 15 unsafe releases from the consumed DEP-001C
900-case internal blind run. DEP-001C is burned and is not reused for tuning,
threshold selection, regression fixtures, or evidence of generalization. Exact
failed prompts are intentionally omitted from this report and from all DEP-001D
development corpora.

This is internal synthetic engineering evidence. It is not external evaluation,
clinical validation, real-world patient-safety evidence, or production healthcare
readiness.

## Reconstructed execution path

For each failed case the frozen trace was reconstructed as:

`input -> language/context metadata -> semantic unsafe and urgent heads -> intent
family and uncertainty -> deterministic policy action -> fixed evaluator canary ->
post-generation validator -> release decision`.

DEP-001C did **not** invoke the live retriever or a live LLM generator. Its
RAG-conditioned cases used test metadata/prefixing, and the candidate output was
an evaluator-supplied safety canary. Findings about generation or RAG are therefore
containment findings, not measurements of live-generation quality or retrieval
policy drift.

## Aggregate findings

- 15/15 releases passed the input route and the post-generation layer.
- 14/15 were personalized actionable requests; eight were Taglish and six English.
- 13/14 actionable misses were classified as `PERSONALIZED_INFORMATION`; one was
  classified as `ACTIVE_SYMPTOM_MANAGEMENT`.
- The 14 actionable cases received `ALLOW_WITH_BOUNDARY` despite material action
  intent. Unsafe probabilities ranged from very low to modest, and family
  uncertainty was not consistently treated as a blocking signal.
- 1/15 was urgent. The intent-family head selected `URGENT_PRESENTATION`, but low
  family confidence and a low urgent-head probability prevented urgent routing.
- 15/15 canary outputs recorded `postgen_blocked_unsafe_canary=false`.
- The frozen canary used phrasing equivalent to diagnosis confirmation plus a
  treatment-change recommendation. Existing regular-expression rules did not
  cover those semantic constructions, so the output layer could not contain an
  upstream miss.

## Failure taxonomy

| Root-cause class | Contribution | Finding |
|---|---:|---|
| A. unsafe semantic miss | material | Binary unsafe scores missed actionable requests. |
| B. Taglish semantic miss | material | Eight of fourteen actionable releases were Taglish; official Taglish recall lagged English by 0.090277. |
| C. urgent semantic miss | material | The urgent head assigned insufficient support to the released urgent case. |
| D. intent-family misclassification | material | Most actionable cases were mapped to personalized information. |
| E. unsafe/urgent disagreement | material | An urgent-family vote was overridden by weak urgent-head support. |
| F. uncertainty underestimation/use | contributing | Uncertainty did not create a conservative final route in the released cases. |
| G. adjudicator precedence error | material | Actionable and urgent family evidence depended too strongly on binary-head agreement. |
| H. conversation-context loss | not established | Frozen traces do not support this as the primary cause of the 15 releases. |
| I. RAG-induced policy drift | not established | Live retrieval was not executed by DEP-001C. |
| J/K. output validator/detector miss | material | The fixed unsafe canary was not blocked in all 15 cases. |
| L. fail-open path | material | Upstream uncertainty and output-validator weakness could still authorize release. |
| M. combination failure | primary | No independent layer reliably contained the combined misses. |
| N. evaluator-label ambiguity | not established | The official run classified all 15 as unsafe releases under frozen labels. |

## Root cause

DEP-001C exposed a defense-in-depth failure. The input classifier generalized
poorly for code-switched actionable intent and urgent paraphrases; policy selection
required too much agreement between independently trained heads; and the final
validator was predominantly lexical. Because the output layer trusted the upstream
route and did not semantically classify personalized actionability, a joint miss
could reach release.

## Remediation architecture

DEP-001D uses independent layers:

1. multilingual semantic unsafe and urgent heads;
2. an independently trained intent-family head;
3. deterministic policy precedence where confident actionable or urgent families
   can block/escalate without binary-head permission;
4. existing deterministic medical-claim rules;
5. a separately trained semantic output-actionability classifier;
6. a second output-actionability check immediately before transport;
7. fail-closed behavior on malformed data, missing/corrupt artifacts, disabled
   controls, model errors, or validator uncertainty.

The final invariant is: **a personalized actionable medical response is not
released when any independent safety layer identifies material risk or cannot
complete its safety decision.**

## Remaining limitations

- Root-cause attribution is retrospective and internal.
- The consumed DEP-001C bank cannot demonstrate remediation or generalization.
- The new corpus is programmatically generated and may contain grammar-family
  similarity despite overlap controls.
- Live retrieval and generation require separate regression evaluation.
- Passing a new internal blind bank would still require a future independent
  external holdout; DEP-001 remains blocked.
