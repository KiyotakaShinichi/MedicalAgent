# Limitations

OncoTrack is an engineering prototype. Its strongest claims are about software,
safety scaffolding, reproducibility, and synthetic-data MLE discipline.

## Current Hard Limits

- No real patient data is used.
- No model is clinically validated.
- No clinician, oncology nurse, pharmacist, or genetic counselor has signed off
  on the rules or language yet.
- The medical KB is source-backed but not a licensed professional guideline
  corpus.
- Synthetic model performance can reflect generator shortcuts.
- RAG grounding metrics are engineering proxies.
- Optional NLI/entailment validation is not a hard dependency.

## Practical Consequences

- Model scores must be described as monitoring-only synthetic signals.
- RAG answers must be described as patient education and safety routing only.
- Genetic, biomarker, and tumor-marker modules organize records; they do not
  interpret results.
- Supplement content flags interaction review; it does not clear products as
  safe.

## Near-Term Work That Does Not Require Paid Data

- Expand curated KB coverage.
- Keep release gates green.
- Tighten trace replay and claim validation.
- Improve docs and proof packaging.
- Refactor oversized modules.
- Prepare advisor-review packets and rubrics.

## Work That Requires Future Access

- Clinician review.
- Real de-identified cohort validation.
- Prospective shadow-mode evaluation.
- Regulated product quality management.
- Production-grade privacy/security review.
