# Fine-Tune Semantic Contamination Screen

- Status: `needs_attention`
- Screening method: `word_and_character_tfidf_cosine_proxy`
- Flagged pairs: `150`
- Flag rows retained in JSON: `150`
- Flag rows omitted by artifact cap: `0`
- Critical pairs: `7`
- Unresolved pairs: `150`
- Human review completed: `False`
- Cleared for candidate comparison: `False`

Flagged rows retain IDs, source paths, and similarity scores only. The report deliberately omits prompt and answer text.

## Interpretation

This screen catches lexical and near-lexical overlap. It can miss low-overlap paraphrases and can over-flag shared safety language. Every flag must be adjudicated before an offline adapter can advance.

## Boundary

TF-IDF similarity is a lexical-semantic screening proxy, not a clinical semantic model and not proof that contamination is absent. Flagged pairs require human adjudication before any offline adapter promotion.
