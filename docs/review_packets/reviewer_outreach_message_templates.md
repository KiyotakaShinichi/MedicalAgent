# Reviewer outreach message templates

> **Use as-is or adapt lightly.** Every template explicitly states the
> project is a **student engineering prototype, not clinically
> validated**, and that **review does not imply approval**. Do not
> remove those lines. Do not promise anything in exchange for the
> review except a written acknowledgement of the reviewer's role
> (no real name; role descriptor only).

Five role-specific templates follow. Each is short (≤ 200 words) so a
busy person reads it in under a minute.

---

## 1. External peer engineer

> Subject: 30-min review of a student RAG eval set — engineering only, not clinical
>
> Hi {name},
>
> I'm building **NLCare / MedicalAgent**, a student engineering
> prototype of a safety-first, non-diagnostic breast cancer
> monitoring agent. It is **not clinically validated**, has no
> clinician sign-off, and uses synthetic-only data. I would value
> 30–60 minutes of your time to author a small held-out RAG eval set
> under a **no-read protocol** (you don't read my goldset, alias
> map, or failure analyses before writing cases).
>
> Concretely: you'd write 9–15 patient-style queries in a JSONL
> template, pick canonical source IDs, and commit them under a
> short attestation. There's nothing to install. Full instructions:
> [`docs/evals/no_read_rag_goldset_protocol.md`].
>
> This is unpaid, informal, and **does not imply your endorsement,
> approval, or any clinical validation** — your name does not need
> to appear; a role descriptor is fine.
>
> No worries at all if it's not a good fit; please reply "pass" and
> I'll move on without follow-up.
>
> Thanks for considering it,
> {your name}

---

## 2. Senior MLE / AI engineer

> Subject: 60-min senior MLE / RAG eval review — engineering-only, not clinical
>
> Hi {name},
>
> I'm finishing **NLCare / MedicalAgent**, a student engineering
> prototype. It uses synthetic-only data and **is not clinically
> validated**. The RAG/MLE evaluation surface has grown to ~126
> release-gate artifacts, and I'd benefit from a senior pair of
> eyes for 45–60 minutes on the credibility framing — specifically:
>
> - leakage audit, patient-level temporal CV, calibration dossier,
>   conformal coverage, subgroup metrics;
> - source-governed RAG vs BM25 (currently
>   `improvement_proven_vs_bm25 = false`);
> - held-out adversarial generalisation gap (~0.06).
>
> Packet: [`docs/review_packets/senior_mle_eval_review_packet.md`].
> Read time ~45 min; written feedback in a markdown file or a
> quick call — your preference.
>
> This is unpaid, informal, and **does not imply endorsement,
> approval, or clinical validation**. Role descriptor only (no
> real name) is fine.
>
> If it's not a good fit, please reply "pass" — no follow-up.
>
> Thanks,
> {your name}

---

## 3. Oncology nurse / oncology resident

> Subject: 45-min review of patient-facing refusal wording — student prototype, NOT for patient care
>
> Hi {name},
>
> I'm building a student engineering prototype called **NLCare /
> MedicalAgent**. It is a non-diagnostic monitoring agent on
> synthetic data — **not clinically validated, not for patient
> care, no clinician sign-off**. I'd value 45–60 minutes of your
> oncology nursing/clinical perspective on the **patient-facing
> wording** of the agent's refusal templates, urgent-symptom
> triggers, and special-population handling.
>
> Concretely: read a short packet
> ([`docs/review_packets/nurse_or_clinician_safety_review_packet.md`]),
> mark up wording you find unclear, alarming, or
> clinically-incorrect-tone, and write a few bullets in a markdown
> file. There is no patient data — no PHI — and you are not being
> asked to validate any clinical content.
>
> This is unpaid, informal, and **does not imply endorsement, approval,
> or clinical sign-off**.
> Role descriptor only (e.g. "oncology nurse, NN years experience")
> is fine; no real name needed.
>
> If you don't have bandwidth, please just reply "pass" and I'll
> move on without follow-up.
>
> Thank you for considering it,
> {your name}

---

## 4. Genetic counselor

> Subject: 30-min review of VUS / genetic-risk patient wording — student prototype, NOT clinical advice
>
> Hi {name},
>
> I'm building a student engineering prototype called **NLCare /
> MedicalAgent**. It is non-diagnostic, **not clinically validated**,
> and uses synthetic-only data. The agent has a hand-coded
> Variant-of-Uncertain-Significance (VUS) handling layer and I'd
> value 30–45 minutes of your perspective on the **patient-facing
> wording** of how VUS is explained / declined.
>
> Concretely: read a short packet
> ([`docs/review_packets/genetic_counselor_vus_review_packet.md`]),
> highlight wording that misrepresents VUS, suggest safer phrasing,
> and write a short bullet list of concerns. There is no patient
> data — no PHI. You are not being asked to validate genetic
> interpretations; only the patient-facing wording.
>
> Unpaid, informal, and **does not imply endorsement, approval, or
> clinical sign-off**. A role descriptor (e.g. "board-certified
> genetic counselor, oncology focus") is fine; no real name needed.
>
> No worries if it's not a good fit — reply "pass" and I'll move on
> without follow-up.
>
> Thanks for considering it,
> {your name}

---

## 5. Patient advocate / healthcare usability reviewer

> Subject: 30-min usability check on a patient-facing dashboard — engineering prototype, NOT clinical
>
> Hi {name},
>
> I'm building **NLCare / MedicalAgent**, a student engineering
> prototype of a patient-facing monitoring dashboard on synthetic
> data. It is **not clinically validated**, **not for patient
> care**, and has no clinician sign-off. I would benefit from
> 30–45 minutes of patient-advocate or healthcare-usability
> perspective on **how a non-clinical reader experiences the UI** —
> specifically:
>
> - Is the "engineering prototype only" banner clear or banner-blind?
> - Does the dashboard tempt overtrust despite the disclaimers?
> - Is the refusal language compassionate or robotic?
>
> No patient data — no PHI. I can send a screencast or share a
> read-only demo URL; happy to chat or take written notes — your
> preference.
>
> Unpaid, informal, **does not imply endorsement, approval, or
> clinical validation**. Role descriptor only (no real name).
>
> If it's not a good fit, please reply "pass" — no follow-up.
>
> Thanks for considering it,
> {your name}

---

## What every template includes (test-enforced)

The test suite asserts that every template contains:

- "not clinically validated" or "not clinical validation" (verbatim).
- "does not imply" + "approval" (in the same paragraph).
- A clear opt-out line ("reply 'pass'" or equivalent).
- A time budget (30 / 45 / 60 minutes).
- A clear "no patient data / no PHI" sentence where relevant.
- The word "unpaid".

The test suite asserts that NO template contains the banned-phrase
list documented in
[`docs/portfolio_safe_wording.md`](../portfolio_safe_wording.md) as
bare claims (i.e. not in negation). The portfolio doc owns the
canonical banned list; this doc inherits it.

## What to do AFTER you get a "yes"

1. Send the reviewer the matching packet from
   [`docs/review_packets/INDEX.md`](INDEX.md).
2. Ask them to file an attestation in
   `Data/evals/external_review/` using
   [`reviewer_attestation_template.md`](../../Data/evals/external_review/reviewer_attestation_template.md).
3. Ask them to file feedback in
   [`reviewer_feedback_template.csv`](../../Data/evals/external_review/reviewer_feedback_template.csv)
   (or in the artifact-specific review packet).
4. Run
   `python scripts/run_external_review_execution_readiness.py`
   to refresh the readiness artifact.
5. **Do not** mark any review as completed until the attestation
   and the feedback file are both committed.
