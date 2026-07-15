# Full Potential Under Current Constraints

NLCare can be excellent as a student-built, synthetic-only healthcare AI
engineering prototype without pretending to be clinically validated. This file
defines what "10/10 under current constraints" means.

The constraints are explicit:

- no real patient data
- no real-world clinical validation
- no paid clinician, oncology nurse, pharmacist, or genetic counselor review yet
- no diagnosis, treatment recommendation, prognosis, dosage instruction,
  genetic-risk interpretation, tumor-marker interpretation, or supplement
  safety approval

## 1. AI / RAG / Agent Layer

10/10 under current constraints means the agent is behaviorally controlled,
observable, and benchmarked.

Required standard:

- safety routing before generation
- post-generation validation after generation
- source-governed RAG with T1-T5 tiers, allowed use, and staleness checks
- claim-level citation validation
- refusal and insufficient evidence treated as successful outcomes when the
  user asks for unsafe or unsupported claims
- live-agent benchmarks, not only deterministic stubs
- Taglish/code-switched safety parity
- trace replay for each RAG answer: user question, intent, safety gate, rewrite,
  retrieved docs, source tiers, claim validation, post-generation validation,
  and final answer
- no diagnosis, treatment, prognosis, dosage, genetic-risk, or tumor-marker
  overclaims

What can be achieved now:

- robust safety benchmarks
- source governance
- citation and claim checks
- replayable traces
- multilingual safety regression
- optional NLI/entailment as a stronger local mode with heuristic fallback

What must wait:

- clinician-approved medical response templates
- real patient-facing deployment
- validated clinical safety claims

## 2. ML / MLE Layer

10/10 under current constraints means the ML system is honest, leakage-aware,
calibrated as an engineering exercise, and reproducible.

Required standard:

- synthetic-only claim boundary on every model artifact
- leakage-proof training pipeline
- unified promotion/release gate
- modality-aware inference
- evidence-aware abstention
- per-head calibration story
- counterfactual stability tests
- shortcut audits
- synthetic generator cards
- prediction traceability
- no promotion without temporal and external validation

What can be achieved now:

- strong synthetic governance
- leakage CI gates
- shortcut disclosures
- modality-dropout robustness
- abstention behavior
- traceable prediction envelopes
- feature-removal and single-feature audits

What must wait:

- claims of real treatment-response prediction
- real calibration on clinical cohorts
- real fairness/subgroup conclusions
- external clinical endpoint validation

## 3. Software Engineering Layer

10/10 under current constraints means the project is boringly reproducible.

Required standard:

- no red tests before done
- `make ship` and `python scripts/ship.py`
- pre-commit integration gate for `tests/test_breast_monitoring.py`
- split god modules over time
- generated OpenAPI frontend types where practical
- normalized endpoint shapes for benchmark surfaces
- local DB/artifacts not incorrectly committed
- benchmark freshness checks
- CI blocks stale, missing, or failed critical artifacts

What can be achieved now:

- cross-platform local ship gate
- GitHub Actions quality gates
- release-gate policy as code
- pre-commit gate
- docs that map claims to artifacts and commands

What must wait:

- production auth/RBAC hardening
- deployment-grade observability
- long-term database migration discipline under a real hosted database

## 4. Medical Structure / Safety

10/10 under current constraints means the system is medically structured but
does not claim medical authority.

Required standard:

- clinical ontology/data dictionary
- minimum evidence standards per question type
- medical claim boundary checker
- CTCAE-style hints only as review support, never diagnosis
- drug/supplement safety flags routed to clinician/pharmacist review
- pregnancy, pediatric, fertility, breastfeeding, and survivorship boundaries
- patient-safe language
- clinician-review packet ready
- explicit "not clinical validation" boundary

What can be achieved now:

- structured ontology and allowed values
- safety/refusal templates
- supplement interaction flags
- genetics/VUS/tumor-marker boundaries
- advisor packet and review rubric

What must wait:

- clinician sign-off
- local clinical workflow validation
- institutional privacy/security review
- regulated software pathway analysis beyond documentation

## What This Project Still Cannot Claim

- clinical validation
- diagnostic accuracy
- treatment-response prediction for real patients
- toxicity prediction for real patients
- prognosis or survival estimation
- medication, dose, or treatment recommendations
- genetic counseling or inherited-risk interpretation
- tumor-marker interpretation
- supplement safety clearance

## What Future Real-World Validation Would Require

- clinician/nurse/pharmacist/genetic-counselor review of safety rules
- real de-identified clinical cohort or approved public cohort mapping
- IRB/privacy/legal review if real patient data is used
- external validation with predefined endpoints
- calibration and subgroup evaluation on real cohorts
- prospective shadow-mode monitoring before any clinical use
- formal risk management and regulatory classification work

## Student-Achievable 10/10

As a student-accessible prototype, NLCare can reach 10/10 by making every
controllable claim reproducible:

- every claim maps to an artifact
- every artifact maps to a command
- every command is gated
- every limitation is stated before a reviewer has to ask
- every unsafe request refuses safely
- every model output carries evidence sufficiency and synthetic-only boundary

That is the ceiling we can reach now. Clinical validity is the next mountain,
not something to fake inside the current project.
