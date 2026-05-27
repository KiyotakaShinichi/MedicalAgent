# Metamorphic Safety Eval

Status: internal engineering evidence, not external review.

The metamorphic safety eval generates wording variants from the existing
unsafe-intent families and safe educational negative controls. It checks that
unsafe requests keep the same safe route under common phrasing changes, and
that safe educational questions are not over-refused.

It covers:

- hypothetical phrasing
- "not a doctor, but..." prefaces
- anxious/emotional prefaces
- polite pressure
- Taglish/code-switched prefixes
- spacing noise
- safe education framing

Current output:

```bash
python scripts/run_metamorphic_safety_eval.py
```

Artifact:

```text
Data/evals/safety/latest_metamorphic_safety_eval.json
```

Important limitation: cases are derived from internal prototypes, so this can
catch brittle route logic but cannot replace external-author adversarial cases.
The artifact includes a contamination note for this reason.
