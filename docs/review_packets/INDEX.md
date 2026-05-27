# Review packet index

This folder bundles **time-boxed, role-specific** review packets so a
volunteer reviewer can give useful feedback in **one short sitting**.
Each packet states what to read, what to skip, what to comment on, and
what *not* to be asked (we will not ask any reviewer to sign off on
clinical validity — only to react to phrasing, scope, and structure).

The single most valuable unlock for this project under its current
constraints (synthetic-only, no clinical validation) is **one hour of
a real clinician's attention** on the patient-facing refusal wording
and the genetic-risk/VUS handling. The packets below are sized so
that ask is realistic.

## Packets

| Role | Packet | Time budget | What they react to | What they DO NOT sign off on |
|---|---|---|---|---|
| Oncology nurse / clinician | [nurse_or_clinician_safety_review_packet.md](nurse_or_clinician_safety_review_packet.md) | 45–60 min | Urgent-symptom triggers; patient-facing refusal templates; special-population categories; emotional-distress response modes | Clinical validity; safety of any specific ML output; treatment advice |
| Genetic counselor | [genetic_counselor_vus_review_packet.md](genetic_counselor_vus_review_packet.md) | 30–45 min | VUS-as-positive misinterpretation patterns; genetic-risk overclaim taxonomy; patient-facing wording around BRCA/CHEK2/PALB2 | Validity of any genetic interpretation; specific patient cases |
| Senior MLE | [senior_mle_eval_review_packet.md](senior_mle_eval_review_packet.md) | 60 min | Leakage audit, evidence-aware abstention, patient-level temporal CV, conformal calibration, subgroup metrics | Clinical performance; deployment-readiness |
| External author (adversarial) | [external_author_eval_packet.md](external_author_eval_packet.md) | 45 min | Write fresh adversarial cases without reading existing safety patterns | None — strictly authoring, not judging |
| Agentic-tool reviewer | [agentic_workflow_review_packet.md](agentic_workflow_review_packet.md) | 30 min | Bounded tool actions; confirmation-before-write; forbidden tools | None — bounded tool surface only |

## How to receive feedback safely

1. The reviewer writes comments in a fresh markdown file in
   `Data/evals/external_review/` (create the folder if needed). Format:
   one bullet per concern, prefixed with the section of the packet
   it refers to.
2. The repo owner files the comments as `unresolved_review_comments`
   in the next release-gate run.
3. Each comment is either addressed in a follow-up PR or marked
   `wont_fix_with_rationale`. **No comment is silently dropped.**
4. The reviewer is acknowledged by role only (no PII).

## What we will NEVER ask a reviewer for

- A clinical opinion on a specific patient.
- A sign-off that the system is safe to deploy.
- A promise about any future regulatory path.
- A statement that the project has been clinically validated.

The reviewer's role is to **strengthen the engineering boundaries**,
not to grant them clinical authority.

## Why this index exists

The 2026-05-27 self-critique rated the Medical side 6.5/10 specifically
because no clinician is in the loop. This index is the engineering
move that makes the human-in-loop ask **cheap** when a clinician
volunteer appears — they walk in, pick a packet, finish in <1 hour,
and leave a structured artifact behind.

Until a reviewer engages, the Medical side has a hard ceiling that no
amount of additional code can lift.
