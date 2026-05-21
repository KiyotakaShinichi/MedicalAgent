# Adversarial Generalization

This project tracks adversarial safety as engineering regression evidence, not
clinical safety proof.

The generalization report separates:

- original in-sample bank
- existing held-out variants
- paraphrase robustness set
- safe educational negative controls

Run:

```bash
python scripts/run_adversarial_generalization_eval.py
```

Artifacts:

```text
Data/evals/safety/adversarial_paraphrase_robustness.jsonl
Data/evals/safety/latest_adversarial_generalization_eval.json
```

The paraphrase set is evaluation-only in this pass. If it is used for tuning,
create a newer held-out set before claiming improvement.
