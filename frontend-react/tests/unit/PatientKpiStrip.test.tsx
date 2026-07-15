import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { PatientKpiStrip } from "../../src/pages/patient/PatientKpiStrip";
import type { PatientReport } from "../../src/types/api";


const report = {
  monitoring_score: 42,
  multimodal_assessment: {
    treatment_monitoring_score: 42,
    overall_status: "watch_closely",
    overall_message: "Record review context only.",
    signals: {},
    score_breakdown: {
      base_signal: 76,
      urgent_review_flags: 2,
      urgent_flag_deduction: 24,
      watch_flags: 1,
      watch_flag_deduction: 5,
      peak_recorded_symptom_severity: 4,
      symptom_deduction: 4.8,
      synthetic_lab_provenance_deduction: 0,
      total_deduction: 33.8,
      final_score: 42,
      formula: "clamp(...) ",
      claim_boundary: "Not a clinical prediction.",
    },
    patient_next_steps: ["Review the queue with your care team."],
  },
  latest_labs: { wbc: 3.1, hemoglobin: 10.5, platelets: 120 },
  hybrid_prediction: {
    classification: {
      decision: "favorable_pattern",
      probability: 0.73,
      confidence: "moderate",
      evidence: {
        modalities_present: ["cbc_pre", "symptoms"],
        modalities_missing: ["imaging"],
      },
    },
    toxicity: null,
  },
  ai_summary: { review_reasons: ["One record item needs care-team review."] },
  data_availability: {
    status: "insufficient_data",
    patient_friendly_summary: "Some parts of your record are incomplete.",
    clinician_style_summary: "Engineering summary.",
    fallback_policy: "Do not force a prediction.",
    items: [
      { name: "CBC trend", status: "available", detail: "2 rows", next_step: "" },
      { name: "Imaging trend", status: "missing", detail: "none", next_step: "Add imaging" },
    ],
  },
} as unknown as PatientReport;


describe("PatientKpiStrip metric explanations", () => {
  it("leads with review workload and record coverage instead of a health-like score", async () => {
    render(<PatientKpiStrip report={report} />);

    expect(screen.getByText("Items for review")).toBeInTheDocument();
    expect(screen.queryByText("Stable today")).toBeNull();
    expect(screen.getByText("Synthetic model pattern")).toBeInTheDocument();
    expect(screen.getByText("Record coverage")).toBeInTheDocument();
    expect(screen.queryByText("42")).toBeNull();

    expect(screen.getByText(/1 available record item matched a rule/i)).toBeInTheDocument();
    expect(screen.getByText("Urgent-review rule matches")).toBeInTheDocument();
    expect(screen.getByText("Review the queue with your care team.")).toBeInTheDocument();

    const explainButtons = screen.getAllByRole("button", { name: /explain this indicator/i });
    fireEvent.click(explainButtons[1]);
    expect(screen.getByRole("heading", { name: /synthetic model pattern/i })).toBeInTheDocument();
    expect(screen.getByText(/not the patient's chance of improving/i)).toBeInTheDocument();
  });
});
