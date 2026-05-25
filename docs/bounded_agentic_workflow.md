# Bounded Agentic Workflow Planner

NLCare's next agentic layer is deliberately bounded.  The planner can decide
which workflow path is appropriate, but it does not grant the assistant medical
authority.

Allowed planner roles:

- classify intent and safety boundary
- choose source-backed education
- retrieve sources and validate claims
- prepare clinician-review summaries
- identify structured health-update intent
- ask for confirmation before record writes
- route unsafe requests to refusal or review
- record concise trace reasons

Blocked authority:

- diagnosis
- treatment recommendation
- dosage change
- prognosis or survival estimate
- genetic-risk interpretation
- tumor-marker recurrence/progression conclusion
- supplement replacement advice

Run the internal workflow eval:

```bash
python scripts/build_agentic_workflow_cases.py
python scripts/run_agentic_workflow_eval.py
```

Artifacts:

- `Data/evals/agentic_tool_use/agentic_workflow_cases.jsonl`
- `Data/evals/agentic_tool_use/latest_agentic_workflow_eval.json`

The current planner is an engineering scaffold.  It is not a clinical
workflow engine and is not proof of real patient safety.
