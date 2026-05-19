# Medical structure maturity (synthetic-only)

This page documents the **structural** medical layers OncoTrack ships
today. None of these layers establish clinical validation. They give a
future clinician reviewer a defined surface to inspect.

## What's in place

| Layer | Module | Artifact |
|---|---|---|
| Clinical ontology / data dictionary | `backend/services/clinical_ontology.py` | manifest via `ontology_manifest()` |
| Minimum evidence standards | `backend/services/minimum_evidence.py` | `Data/evals/medical/latest_minimum_evidence_standards.json` |
| Medical claim boundary checker | `backend/services/medical_claim_boundary.py` | `Data/evals/safety/latest_medical_claim_boundary_eval.json` |
| Medical evidence standards | `backend/services/medical_evidence_standards.py` | bundled into advisor packet |
| Medical safety contract | `backend/services/medical_safety_contract.py` | `Data/evals/safety/latest_medical_safety_contract.json` |
| Toxicity-review hints (CTCAE-inspired) | `backend/services/toxicity_review_mapping.py` | bundled into advisor packet |
| Supplement / drug interaction safety | `backend/services/supplement_safety.py` | bundled into advisor packet |
| Special-population boundary handling | `backend/services/special_population_boundary.py` | `tests/test_special_population_boundary.py` (16 tests) |
| Medical advisor packet | `backend/services/medical_advisor_packet.py` | `Data/evals/medical/latest_medical_advisor_review_packet.json` |

## Special-population coverage (PART 1.6)

`special_population_boundary.classify_special_population(query)` detects
the eight categories the spec calls out and returns:

- the `category` (pregnancy, breastfeeding, pediatric, fertility,
  survivorship, recurrence_anxiety, palliative_or_supportive,
  end_of_life_distress)
- `urgent_escalation = True` only for end_of_life_distress
- `matched_terms` (audit trail)
- `safe_wording` (template the answer composition layer can substitute)
- `routing` (oncology_and_obstetrics, pediatric_oncology_team,
  crisis_resources_and_oncology, etc.)

Each safe-wording template includes a clinician routing phrase — the
test suite enforces this.

## What this layer is

A **structure-only** scaffold. It defines:

- which fields a patient can enter,
- which claims an output may or may not contain,
- which evidence is required to attempt a given question type,
- which categories of question must route to a specialist, and
- which questions trigger urgent escalation.

The reviewer-facing field dictionary is also summarized in
`docs/data_dictionary.md`, including the strict common public-bridge fields used
for transfer stress tests.

## What this layer is NOT

- Not clinical validation.
- Not equivalent to a clinician's judgement.
- Not exhaustive — every vocabulary table is heuristic and will miss
  edge cases until a clinician reviewer expands it.
- Not a substitute for the medical advisor packet review, which has
  not yet been performed.

## How to extend safely

1. Add new ontology entries in `clinical_ontology.CLINICAL_ONTOLOGY` —
   always include `blocked_claims` and `allowed_use`.
2. Add new boundary categories in `special_population_boundary` —
   always include a `safe_wording` template with a clinician routing
   phrase.
3. Add new evidence requirements in `minimum_evidence` — always include
   the abstention path.
4. Re-run `python scripts/run_mle_promotion_gate.py` and
   `python scripts/run_offline_ab_eval.py` to verify the change does
   not regress any gate.
