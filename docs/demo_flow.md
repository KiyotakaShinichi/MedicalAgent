# Demo Flow

This demo flow shows how to exercise the safety-first monitoring workflow using the local PoC environment.

## 1. Start the API
```
uvicorn backend.api.main:app --reload
```

## 2. Log in with demo sessions
- Open the login page and select a demo role.
- Patient portal: /patient
- Clinician portal: /clinician
- Admin portal: /admin

## 3. Patient portal flow
- Enter CBC values, symptoms, and medications.
- Upload an imaging report or paste report text.
- Ask a low-risk education question and confirm citations when retrieval context is used.
- Try an urgent or treatment-decision question and confirm refusal or escalation.

## 4. Clinician portal flow
- Open a patient report and review monitoring signals.
- Review timeline summaries and risk flags.
- Submit clinician feedback using approve, edit, reject, or follow-up.

## 5. Admin and MLE flow
- Open the admin dashboard.
- Run the agent regression suite and review guardrail results.
- Run MLE readiness checks and review the readiness status.
- Generate a versioned evaluation report.

## Safety note
All patient-specific or urgent outputs must be reviewed by a qualified clinician. This demo is a PoC and not clinically validated.
