def build_clinical_timeline(labs, treatments, imaging_reports, symptoms, risks):
    events = []

    if labs is not None and not labs.empty:
        for _, row in labs.iterrows():
            events.append({
                "date": str(row["date"]),
                "type": "lab",
                "title": "CBC result",
                "summary": f"WBC {row['wbc']}, hemoglobin {row['hemoglobin']}, platelets {row['platelets']}",
            })

    if treatments is not None and not treatments.empty:
        for _, row in treatments.iterrows():
            events.append({
                "date": str(row["date"]),
                "type": "treatment",
                "title": f"Treatment cycle {row['cycle']}",
                "summary": str(row["drug"]),
            })

    if imaging_reports is not None and not imaging_reports.empty:
        for _, row in imaging_reports.iterrows():
            modality = row.get("modality", "Imaging")
            events.append({
                "date": str(row["date"]),
                "type": "imaging",
                "title": f"{modality} - {row['report_type']}",
                "summary": row["impression"],
            })

    if symptoms is not None and not symptoms.empty:
        for _, row in symptoms.iterrows():
            note = f" - {row['notes']}" if row.get("notes") else ""
            events.append({
                "date": str(row["date"]),
                "type": "symptom",
                "title": f"Symptom: {row['symptom']}",
                "summary": f"Severity {row['severity']}/10{note}",
            })

    for risk in risks:
        evidence = risk.get("evidence") or {}
        risk_date = evidence.get("date")
        if risk_date:
            events.append({
                "date": str(risk_date),
                "type": "risk",
                "title": f"Risk flag: {risk.get('type')}",
                "summary": risk.get("message"),
                "severity": risk.get("severity"),
            })

    return sorted(events, key=lambda event: event["date"])
