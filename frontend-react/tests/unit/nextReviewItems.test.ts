import { describe, it, expect } from "vitest";
import { deriveReviewItems } from "../../src/pages/patient/nextReviewItems";
import type { PatientReport } from "../../src/types/api";

function makeReport(overrides: Partial<PatientReport>): PatientReport {
  // Minimal stub — only fields deriveReviewItems reads need to be present.
  return {
    patient_id: "P1",
    patient_name: "Test Patient",
    diagnosis: "",
    latest_labs: { wbc: null, hemoglobin: null, platelets: null },
    lab_history: [],
    monitoring_score: null,
    overall_status: "stable",
    multimodal_assessment: null,
    synthetic_model_prediction: null,
    synthetic_model_explanation: null,
    ai_summary: null,
    timeline: [],
    treatment_effects: [],
    symptoms: [],
    medication_logs: [],
    chat_history: [],
    uploads: [],
    treatment_outcome: null,
    clinical_interventions: [],
    ...overrides,
  } as PatientReport;
}

describe("deriveReviewItems", () => {
  it("returns an empty list when there is nothing to discuss", () => {
    expect(deriveReviewItems(makeReport({}))).toEqual([]);
  });

  it("surfaces AI review_reasons as warning-toned items", () => {
    const report = makeReport({
      ai_summary: {
        patient_explanation: "",
        clinical_summary: "",
        review_reasons: ["WBC trending lower", "Persistent fatigue"],
      },
    });
    const items = deriveReviewItems(report);
    expect(items).toHaveLength(2);
    expect(items[0]).toMatchObject({
      title: "AI flagged for discussion",
      detail: "WBC trending lower",
      tone: "warning",
    });
    expect(items.every((i) => i.tone === "warning")).toBe(true);
  });

  it("caps review_reasons at 2 even if the model returns more", () => {
    const report = makeReport({
      ai_summary: {
        patient_explanation: "",
        clinical_summary: "",
        review_reasons: ["a", "b", "c", "d", "e"],
      },
    });
    const items = deriveReviewItems(report);
    expect(items.filter((i) => i.title === "AI flagged for discussion")).toHaveLength(2);
  });

  it("includes multimodal signals whose status mentions concern/warning/alert", () => {
    const report = makeReport({
      multimodal_assessment: {
        treatment_monitoring_score: 50,
        overall_status: "watch",
        overall_message: "watch",
        signals: {
          mri_response: { status: "concerning", message: "Lesion enlarged 12%" },
          clinical_monitoring: { status: "stable", message: "All normal" },
        },
      },
    });
    const items = deriveReviewItems(report);
    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      title: "Mri Response",
      detail: "Lesion enlarged 12%",
      tone: "warning",
    });
  });

  it("ignores low-severity timeline events but keeps moderate+ events", () => {
    const report = makeReport({
      timeline: [
        { date: "2026-05-01", type: "lab",     severity: "low",      title: "WBC normal",   summary: "normal" },
        { date: "2026-05-10", type: "symptom", severity: "moderate", title: "Mild nausea",  summary: "after cycle 3" },
        { date: "2026-05-12", type: "imaging", severity: "high",     title: "MRI concern",  summary: "lesion update" },
      ],
    });
    const items = deriveReviewItems(report);
    // Sorted by date desc; high event tone is "warning", moderate is "info"
    expect(items).toHaveLength(2);
    expect(items[0]).toMatchObject({ title: "MRI concern",  tone: "warning" });
    expect(items[1]).toMatchObject({ title: "Mild nausea",  tone: "info" });
  });

  it("deduplicates and caps the final list at 4 items", () => {
    const repeated = "AI flagged for discussion";
    const report = makeReport({
      ai_summary: {
        patient_explanation: "",
        clinical_summary: "",
        review_reasons: ["same", "same"],
      },
      multimodal_assessment: {
        treatment_monitoring_score: 50,
        overall_status: "watch",
        overall_message: "",
        signals: {
          a: { status: "concerning", message: "m1" },
          b: { status: "warning",    message: "m2" },
          c: { status: "alert",      message: "m3" },
          d: { status: "concerning", message: "m4" },
        },
      },
    });
    const items = deriveReviewItems(report);
    expect(items.length).toBeLessThanOrEqual(4);
    expect(items.filter((i) => i.title === repeated)).toHaveLength(1);
  });
});
