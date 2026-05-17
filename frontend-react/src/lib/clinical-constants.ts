/**
 * Central, single-source-of-truth configuration for all clinical reference
 * ranges, units, status thresholds, and patient-safe explanatory copy.
 *
 * Why this file exists
 * --------------------
 * Reference ranges are not constants you should let drift into individual
 * components.  Lab card renderers, the chat-parsed CBC save flow, the
 * education tooltips, and any future clinician-facing summary all need the
 * SAME numbers.  If the ranges live in this file:
 *   - one diff updates every reading-surface at once,
 *   - the chat agent and the UI cannot disagree about what counts as
 *     "borderline",
 *   - a clinical advisor can review one file and sign off on every range.
 *
 * Important non-claims
 * --------------------
 *  - These are population-level adult oncology-monitoring reference ranges.
 *    Real laboratories publish their own, sex/age/assay-specific ranges that
 *    take precedence.  Treat values here as **default** ranges only.
 *  - "Critical" thresholds are SAFETY signals for the UI — not clinical
 *    intervention thresholds.  They exist so the patient sees a calm-but-
 *    visible warning chip; they are not an order to do anything.
 *  - These numbers are PoC defaults.  Production deployments should override
 *    via a config endpoint or a clinician-curated lookup.
 */

export type LabStatus =
  | "critical_low"
  | "low"
  | "borderline"
  | "in_range"
  | "high"
  | "critical_high"
  | "unknown";

/** Tone hint for badges + lab-card left bars. */
export type LabStatusTone = "danger" | "warning" | "success" | "neutral";

export interface LabReferenceRange {
  /** Display name shown on cards. */
  label: string;
  /** Unit string shown next to the value (kept verbatim — no parsing). */
  unit: string;
  /** Inclusive lower bound of the in-range band. */
  refLow: number;
  /** Inclusive upper bound of the in-range band. */
  refHigh: number;
  /** Below this, we surface a "critical low" warning chip. */
  criticalLow: number;
  /** Above this, we surface a "critical high" warning chip. */
  criticalHigh: number;
  /** Patient-friendly explanation shown in the lab card tooltip. */
  description: string;
  /** Required disclaimer that travels with every reading-surface. */
  disclaimer: string;
}

/**
 * Default CBC reference ranges used by patient-facing lab cards.
 * Values reflect commonly-cited adult oncology-monitoring ranges and are
 * deliberately conservative — real labs publish their own narrower ranges.
 */
export const LAB_REFERENCE_RANGES = {
  wbc: {
    label: "WBC",
    unit: "K/uL",
    refLow: 4.0,
    refHigh: 11.0,
    criticalLow: 1.0,
    criticalHigh: 30.0,
    description:
      "White blood cells help the body respond to infection. Low values can happen during chemotherapy and may need clinician review.",
    disclaimer:
      "Reference ranges vary by lab and by patient context. Treat this as a guide, not a diagnosis.",
  },
  hemoglobin: {
    label: "Hemoglobin",
    unit: "g/dL",
    refLow: 12.0,
    refHigh: 16.0,
    criticalLow: 7.0,
    criticalHigh: 20.0,
    description:
      "Hemoglobin carries oxygen in red blood cells. Lower values can cause fatigue and may need follow-up with your care team.",
    disclaimer:
      "Reference ranges vary by lab and by patient context. Sex-specific ranges should be used when available.",
  },
  platelets: {
    label: "Platelets",
    unit: "K/uL",
    refLow: 150.0,
    refHigh: 400.0,
    criticalLow: 50.0,
    criticalHigh: 1000.0,
    description:
      "Platelets help blood clot. Low values can increase bleeding risk and should be reviewed by your care team.",
    disclaimer:
      "Reference ranges vary by lab and by patient context. Treat this as a guide, not a diagnosis.",
  },
  anc: {
    label: "ANC",
    unit: "K/uL",
    refLow: 1.5,
    refHigh: 8.0,
    criticalLow: 0.5,
    criticalHigh: 15.0,
    description:
      "Absolute neutrophil count measures infection-fighting cells. Very low ANC during chemotherapy is a clinical concern.",
    disclaimer:
      "ANC thresholds for febrile neutropenia are clinician-managed; this card is a signal only.",
  },
} satisfies Record<string, LabReferenceRange>;

export type LabKey = keyof typeof LAB_REFERENCE_RANGES;

/** Classify a single numeric value against its lab's reference range. */
export function classifyLabValue(key: LabKey, value: number | null | undefined): LabStatus {
  if (value == null || !Number.isFinite(value)) return "unknown";
  const range = LAB_REFERENCE_RANGES[key];
  if (value <= range.criticalLow)  return "critical_low";
  if (value >= range.criticalHigh) return "critical_high";
  if (value < range.refLow) {
    // Borderline if within 10% below the lower bound, otherwise plainly low.
    const margin = (range.refLow - value) / range.refLow;
    return margin <= 0.10 ? "borderline" : "low";
  }
  if (value > range.refHigh) {
    const margin = (value - range.refHigh) / range.refHigh;
    return margin <= 0.10 ? "borderline" : "high";
  }
  return "in_range";
}

/** Map a lab status to a UI badge tone (consumed by ``StatusBadge``). */
export function labStatusTone(status: LabStatus): LabStatusTone {
  switch (status) {
    case "critical_low":
    case "critical_high":
      return "danger";
    case "low":
    case "high":
    case "borderline":
      return "warning";
    case "in_range":
      return "success";
    case "unknown":
    default:
      return "neutral";
  }
}

/** Short, patient-safe label for a lab status. */
export function labStatusLabel(status: LabStatus): string {
  switch (status) {
    case "critical_low":  return "Very low";
    case "low":           return "Low";
    case "borderline":    return "Borderline";
    case "in_range":      return "In range";
    case "high":          return "High";
    case "critical_high": return "Very high";
    case "unknown":
    default:              return "No value";
  }
}

/**
 * Allowed imaging modalities for the manual-imaging-save form.
 * Kept here so the chat-agent intent classifier, the upload-classification
 * pipeline (when it lands), and the patient form share the same vocabulary.
 */
export const IMAGING_MODALITIES = [
  { value: "MRI",        label: "MRI" },
  { value: "CT",         label: "CT" },
  { value: "Ultrasound", label: "Ultrasound" },
  { value: "Mammogram",  label: "Mammogram" },
  { value: "Other",      label: "Other" },
] as const;

export type ImagingModality = (typeof IMAGING_MODALITIES)[number]["value"];

/**
 * Symptom severity scale → human-readable bucket.
 * The patient enters 0–10 on a slider; this bucket drives the colour and the
 * "urgent red flag" prompt in the symptom form.
 */
export function severityBucket(severity: number): "mild" | "moderate" | "severe" {
  if (severity <= 3) return "mild";
  if (severity <= 6) return "moderate";
  return "severe";
}

/**
 * Universal safety disclaimer that goes at the bottom of every clinical
 * input form.  Imported by every form so wording stays consistent.
 */
export const NON_DIAGNOSTIC_DISCLAIMER =
  "This portal records what you enter — it does not diagnose, recommend treatment, or replace clinician judgement. Discuss anything concerning with your care team.";

/**
 * Curated catalog of common breast-cancer-treatment symptoms.  Used by the
 * SymptomForm dropdown to reduce typo/spelling drift in saved patient data
 * while still allowing free text via the "Other" fallback.
 *
 * Aligned with the kinds of adverse events commonly tracked during
 * chemotherapy + targeted therapy monitoring (loosely CTCAE-aligned, but
 * the wording is patient-facing, not clinician-graded).  Add new entries
 * here rather than letting them drift into individual forms.
 */
export const COMMON_SYMPTOMS = [
  { value: "Fatigue",            label: "Fatigue (tiredness)" },
  { value: "Nausea",             label: "Nausea" },
  { value: "Vomiting",           label: "Vomiting" },
  { value: "Fever",              label: "Fever" },
  { value: "Chills",             label: "Chills" },
  { value: "Diarrhea",           label: "Diarrhea" },
  { value: "Constipation",       label: "Constipation" },
  { value: "Mouth sores",        label: "Mouth sores (mucositis)" },
  { value: "Loss of appetite",   label: "Loss of appetite" },
  { value: "Taste changes",      label: "Taste changes" },
  { value: "Hair loss",          label: "Hair loss" },
  { value: "Skin reaction",      label: "Skin reaction or rash" },
  { value: "Nail changes",       label: "Nail changes" },
  { value: "Neuropathy",         label: "Neuropathy (numbness/tingling in hands or feet)" },
  { value: "Joint pain",         label: "Joint pain" },
  { value: "Muscle pain",        label: "Muscle pain" },
  { value: "Headache",           label: "Headache" },
  { value: "Shortness of breath",label: "Shortness of breath" },
  { value: "Chest pain",         label: "Chest pain" },
  { value: "Heart palpitations", label: "Heart palpitations" },
  { value: "Easy bruising",      label: "Easy bruising or bleeding" },
  { value: "Hot flashes",        label: "Hot flashes" },
  { value: "Sleep problems",     label: "Sleep problems" },
  { value: "Anxiety",            label: "Anxiety" },
  { value: "Depression",         label: "Depression or low mood" },
  { value: "Brain fog",          label: "Brain fog (memory or focus problems)" },
  { value: "Lymphedema",         label: "Swelling (lymphedema)" },
  { value: "Breast pain",        label: "Breast or chest-wall pain" },
] as const;

export type CommonSymptomValue = (typeof COMMON_SYMPTOMS)[number]["value"];

/**
 * Curated catalog of common medications used during breast cancer treatment
 * monitoring.  Grouped narratively for the dropdown:
 *
 *   - Chemotherapy backbone (anthracyclines + taxanes + alkylators)
 *   - Targeted therapy (HER2-directed)
 *   - Endocrine therapy (ER+ patients)
 *   - Supportive care (antiemetics, growth factors, steroids, etc.)
 *
 * Real prescription decisions belong to the clinician — this is purely a
 * data-entry hint, with an "Other (specify)" fallback so a patient can still
 * enter anything their care team has them on.
 */
export const COMMON_MEDICATIONS = [
  // Chemotherapy backbone
  { value: "Doxorubicin",      label: "Doxorubicin (Adriamycin)", group: "Chemotherapy" },
  { value: "Epirubicin",       label: "Epirubicin",                group: "Chemotherapy" },
  { value: "Cyclophosphamide", label: "Cyclophosphamide (Cytoxan)", group: "Chemotherapy" },
  { value: "Paclitaxel",       label: "Paclitaxel (Taxol)",        group: "Chemotherapy" },
  { value: "Docetaxel",        label: "Docetaxel (Taxotere)",      group: "Chemotherapy" },
  { value: "Carboplatin",      label: "Carboplatin",                group: "Chemotherapy" },
  { value: "5-Fluorouracil",   label: "5-Fluorouracil (5-FU)",     group: "Chemotherapy" },
  { value: "Capecitabine",     label: "Capecitabine (Xeloda)",     group: "Chemotherapy" },
  // Targeted (HER2)
  { value: "Trastuzumab",      label: "Trastuzumab (Herceptin)",   group: "Targeted therapy" },
  { value: "Pertuzumab",       label: "Pertuzumab (Perjeta)",      group: "Targeted therapy" },
  { value: "T-DM1",            label: "T-DM1 (Kadcyla)",           group: "Targeted therapy" },
  // Endocrine
  { value: "Tamoxifen",        label: "Tamoxifen",                  group: "Endocrine therapy" },
  { value: "Anastrozole",      label: "Anastrozole (Arimidex)",    group: "Endocrine therapy" },
  { value: "Letrozole",        label: "Letrozole (Femara)",        group: "Endocrine therapy" },
  { value: "Exemestane",       label: "Exemestane (Aromasin)",     group: "Endocrine therapy" },
  // Supportive care
  { value: "Ondansetron",      label: "Ondansetron (Zofran)",      group: "Supportive (antiemetic)" },
  { value: "Granisetron",      label: "Granisetron (Kytril)",      group: "Supportive (antiemetic)" },
  { value: "Dexamethasone",    label: "Dexamethasone",              group: "Supportive (steroid)" },
  { value: "Filgrastim",       label: "Filgrastim (Neupogen) — G-CSF", group: "Supportive (growth factor)" },
  { value: "Pegfilgrastim",    label: "Pegfilgrastim (Neulasta) — long-acting G-CSF", group: "Supportive (growth factor)" },
  { value: "Loperamide",       label: "Loperamide (Imodium)",      group: "Supportive (anti-diarrheal)" },
  { value: "Paracetamol",      label: "Paracetamol (acetaminophen)", group: "Supportive (analgesic)" },
  { value: "Ibuprofen",        label: "Ibuprofen",                  group: "Supportive (analgesic)" },
] as const;

export type CommonMedicationValue = (typeof COMMON_MEDICATIONS)[number]["value"];

/** Sentinel value the SelectWithCustom helper uses for the "Other" branch. */
export const OTHER_OPTION_VALUE = "__other__";
