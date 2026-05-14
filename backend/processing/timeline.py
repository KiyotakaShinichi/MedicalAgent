"""Patient timeline assembly.

The timeline merges lab CBC entries, treatment cycles, imaging report
summaries, symptom reports, and AI risk flags into a chronologically
ordered list of events for the patient view.

Risk events (produced by the deterministic risk engine) are tagged as
``ai_generated`` and carry the uncertainty block from the risk evidence
so the frontend ``AIGeneratedLabel`` can render confidence, uncertainty
reason, missing-data indicators, and the "Clinician review required"
amber badge. See [SAFETY_CARD.md](../../SAFETY_CARD.md) for why every
AI/model output must surface these fields together.
"""


def build_clinical_timeline(labs, treatments, imaging_reports, symptoms, risks):
    events = []
    seen = set()

    def add_event(event):
        key = (
            str(event.get("date", ""))[:10],
            event.get("type"),
            event.get("title"),
            event.get("summary"),
        )
        if key in seen:
            return
        seen.add(key)
        events.append(event)

    if labs is not None and not labs.empty:
        for _, row in labs.iterrows():
            add_event({
                "date": str(row["date"]),
                "type": "lab",
                "title": "CBC result",
                "summary": (
                    f"WBC {row['wbc']}, hemoglobin {row['hemoglobin']}, "
                    f"platelets {row['platelets']}"
                ),
                "ai_generated": False,
                "evidence_source": "lab_record",
            })

    if treatments is not None and not treatments.empty:
        for _, row in treatments.iterrows():
            add_event({
                "date": str(row["date"]),
                "type": "treatment",
                "title": f"Treatment cycle {row['cycle']}",
                "summary": str(row["drug"]),
                "ai_generated": False,
                "evidence_source": "treatment_record",
            })

    if imaging_reports is not None and not imaging_reports.empty:
        for _, row in imaging_reports.iterrows():
            modality = row.get("modality", "Imaging")
            add_event({
                "date": str(row["date"]),
                "type": "imaging",
                "title": f"{modality} - {row['report_type']}",
                "summary": row["impression"],
                "ai_generated": False,
                "evidence_source": "imaging_report",
            })

    if symptoms is not None and not symptoms.empty:
        for _, row in symptoms.iterrows():
            note = f" - {row['notes']}" if row.get("notes") else ""
            add_event({
                "date": str(row["date"]),
                "type": "symptom",
                "title": f"Symptom: {row['symptom']}",
                "summary": f"Severity {row['severity']}/10{note}",
                "ai_generated": False,
                "evidence_source": "patient_report",
            })

    for risk in risks:
        evidence = risk.get("evidence") or {}
        risk_date = evidence.get("date")
        if risk_date:
            add_event({
                "date": str(risk_date),
                "type": "ai_risk_flag",
                "title": f"Risk flag: {risk.get('type')}",
                "summary": risk.get("message"),
                "severity": risk.get("severity"),
                "ai_generated": True,
                "evidence_source": "risk_engine",
                "model_version": evidence.get("threshold_config_version"),
                "uncertainty": risk.get("uncertainty"),
            })

    return sorted(events, key=lambda event: event["date"])
