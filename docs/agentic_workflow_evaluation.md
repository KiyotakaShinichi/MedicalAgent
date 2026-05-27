# Agentic Workflow Evaluation

NLCare uses a bounded planner-executor-verifier scaffold for agentic behavior. The goal is not to make the system medically autonomous. The goal is to make each patient-support turn choose an auditable route, restrict tool use, require confirmation before writes, and verify that the final response stayed inside the non-diagnostic claim boundary.

## Contract

Each turn is split into:

- Planner: classifies the route, allowed tools, blocked tools, confirmation requirement, and review route.
- Executor: simulates only tools allowed by the plan. Write tools are skipped unless the caller marks the turn as confirmed by the user.
- Verifier: checks that forbidden tools were not executed, unplanned tools were not executed, write tools were not run without confirmation, source-backed education used retrieval/claim validation, and refusal responses did not keep misleading citations.
- Response packager: returns a safe refusal, source-backed education stub, missing-details request, confirmation prompt, saved-record acknowledgement, or clinician-review summary.

Forbidden medical authority remains blocked:

- diagnosis
- treatment recommendation
- dose change
- prognosis or survival estimate
- genetic-risk interpretation
- tumor-marker recurrence conclusion

## Evaluations

Artifacts:

- `Data/evals/agentic_tool_use/latest_agentic_workflow_eval.json`
- `Data/evals/agentic_tool_use/latest_agentic_tool_use_eval.json`
- `Data/evals/agentic_tool_use/latest_multiturn_agent_eval.json`
- `Data/evals/agentic_tool_use/latest_adversarial_tool_use_eval.json`

The evals cover safe education, structured record capture, missing-detail requests, confirmation-before-write, multi-turn symptom completion, prompt injection, cross-patient requests, genetics/VUS boundaries, tumor-marker boundaries, treatment/dosage boundaries, supplement replacement, and urgent symptom minimization.

## Current Boundary

This is engineering evidence only. It shows the workflow contract can prevent unsafe tool execution under curated internal cases. It does not prove clinical safety, real-world robustness, clinician approval, or that future external-author adversarial cases will pass.

Next evidence upgrades:

- external-author tool-use cases
- live UI-to-backend tool-use traces
- larger multi-turn holdouts
- audited failure reviews by a senior MLE or clinician reviewer
