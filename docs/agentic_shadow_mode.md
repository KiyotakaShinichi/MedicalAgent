# Agentic Shadow Mode

Agentic shadow mode compares the bounded planner with the current orchestrated turn path on curated internal cases.

Run:

```bash
python scripts/run_agentic_shadow_mode_eval.py
```

Output:

```text
Data/evals/agentic_tool_use/latest_agentic_shadow_mode_eval.json
```

The eval checks:

- route agreement between planner and orchestrator
- forbidden tool leakage
- unsafe write leakage
- trace reasons for route decisions

Boundary: this is shadow-mode engineering evidence only. It does not prove autonomous clinical safety or production agent readiness.
