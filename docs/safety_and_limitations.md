# Safety and Limitations

## Safety posture
- Designed with healthcare privacy principles in mind, but not certified or validated for clinical deployment.
- All patient-specific or urgent outputs must be reviewed by a qualified clinician.
- RAG is used for grounded knowledge support, not autonomous medical decision-making.
- ML outputs are monitoring signals and risk flags, not diagnoses.
- Threat model and security controls are documented for PoC review.

## Guardrails
- Deterministic scope and safety checks for urgent, diagnosis, and treatment-decision requests. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)
- Prompt-injection and privacy-boundary detection with multilingual variants. Evidence: [backend/services/security_guardrails.py](backend/services/security_guardrails.py)
- Output validation for treatment directives, diagnosis claims, and missing citations. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)
- Refusal and escalation to clinician review for unsafe or urgent requests. Evidence: [backend/services/agent_rag.py](backend/services/agent_rag.py)

## Non-diagnostic boundary
- This system does not diagnose breast cancer.
- This system does not recommend treatment changes.
- This system does not replace clinicians.
- This system is not clinically validated.
- Synthetic data is used for POC workflow and safety testing, not clinical validation.

## Key limitations
- Synthetic data is not clinical evidence and does not prove real-world performance.
- RAG grounding metrics are heuristic proxies until labeled KB evaluation data exists.
- Imaging analysis is derived from report text or tabular features, not validated clinical imaging models.
- Privacy and security controls are PoC-grade and not production-certified.
