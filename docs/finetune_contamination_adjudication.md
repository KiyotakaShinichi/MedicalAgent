# Fine-Tune Contamination Adjudication

The semantic contamination screen produces candidate pairs, not verdicts. The
adjudication packet makes that human decision process resumable and auditable.

Run:

```powershell
python scripts/run_finetune_semantic_contamination.py
python scripts/run_finetune_contamination_adjudication.py
```

Review
`Data/finetune/evaluations/semantic_contamination_adjudication_packet.json`.
For every candidate, select `contaminated`, `not_contaminated`, or
`ambiguous`, and provide reviewer role, review date, and rationale. The packet
does not copy prompt/response text; the referenced source rows must be
inspected locally.

Completing this internal review does not prove the dataset is uncontaminated.
The TF-IDF screen can miss paraphrases, and an external no-read evaluation is
still required before any credibility claim. Adapter promotion remains
disabled in all cases under the current synthetic-only, clinically unreviewed
boundary.
