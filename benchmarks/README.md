# Benchmark Ladder

This folder captures the portfolio-facing benchmark ladder for MedicalAgent. Every benchmark is engineering evidence, not clinical validation.

Claim boundary:
- Benchmarks measure safety behavior, grounding discipline, and model monitoring readiness.
- Benchmarks do not establish clinical safety, diagnosis performance, or treatment efficacy.

## Files
- safety_eval_cases.jsonl: safety and refusal cases for the benchmark suite.
- rag_eval_cases.jsonl: RAG groundedness and refusal cases for the benchmark suite.
- adversarial_eval_cases.jsonl: adversarial prompt-injection and privacy attack cases.
- clinician_summary_eval_cases.jsonl: rubric cases for clinician-summary benchmarking.
- synthetic_realism_checks.json: realism check definitions for synthetic journeys.
- benchmark_results.csv: generated metrics table (run generator to populate).
- benchmark_report.md: generated markdown report (run generator to populate).

## How to run

Run the individual benchmark suites:

```text
python scripts/run_safety_benchmark.py
python scripts/run_rag_benchmark.py
python scripts/run_adversarial_benchmark.py
python scripts/run_model_benchmark.py
python scripts/run_realism_checks.py
python scripts/run_clinician_summary_benchmark.py
```

Generate the consolidated report:

```text
python scripts/generate_benchmark_report.py
```

Outputs are written into Data/evals/* and summarized in benchmark_results.csv and benchmark_report.md.
