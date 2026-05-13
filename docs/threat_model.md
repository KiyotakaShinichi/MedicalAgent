# Threat Model (PoC)

This threat model documents likely abuse paths for the MedicalAgent PoC and the controls used to reduce risk. It is not a compliance certification.

## Assets
- Patient records and uploads
- Model artifacts and evaluation reports
- Knowledge base sources
- Audit logs and clinician feedback

## Trust boundaries
- Patient and clinician portals to API
- API to database
- API to knowledge base and model artifacts
- Admin/MLE endpoints

## Threat actors
- Curious user attempting to access other patient data
- External attacker trying to exfiltrate data or prompts
- Insider misuse (role escalation, broad data access)

## Top threats
- Prompt injection or jailbreak attempts
- Cross-patient data access
- Database or file system exfiltration
- Role escalation to admin endpoints
- Unsafe medical advice requests

## Current mitigations
- Deterministic guardrails for prompt injection and privacy boundary
- Role-based access checks on admin and patient routes
- Non-diagnostic response policy and refusal templates
- Audit logs for key actions

## Residual risks
- PoC-grade auth and storage controls
- No formal incident response or access reviews
- No production-grade encryption and key management

## Next hardening steps
- Formal threat modeling workshops and abuse-case testing
- Encryption at rest and in transit
- Strong authentication and authorization for all roles
- Formal log retention and access review policies
