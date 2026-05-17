import { describe, it, expect } from "vitest";
import {
  classifyLabValue,
  labStatusTone,
  labStatusLabel,
  severityBucket,
  LAB_REFERENCE_RANGES,
  NON_DIAGNOSTIC_DISCLAIMER,
} from "../../src/lib/clinical-constants";

describe("classifyLabValue", () => {
  it("returns 'unknown' for null/NaN/missing values", () => {
    expect(classifyLabValue("wbc", null)).toBe("unknown");
    expect(classifyLabValue("wbc", undefined)).toBe("unknown");
    expect(classifyLabValue("wbc", Number.NaN)).toBe("unknown");
    expect(classifyLabValue("wbc", Number.POSITIVE_INFINITY)).toBe("unknown");
  });

  it("returns 'in_range' for values inside the band", () => {
    expect(classifyLabValue("wbc", 6.0)).toBe("in_range");
    expect(classifyLabValue("hemoglobin", 13.5)).toBe("in_range");
    expect(classifyLabValue("platelets", 250)).toBe("in_range");
  });

  it("returns 'borderline' for values within 10% of the bound", () => {
    // wbc refLow is 4.0 — 3.7 is 7.5% below, should be borderline
    expect(classifyLabValue("wbc", 3.7)).toBe("borderline");
    // hemoglobin refHigh is 16.0 — 17.0 is 6.25% above, should be borderline
    expect(classifyLabValue("hemoglobin", 17.0)).toBe("borderline");
  });

  it("returns 'low' / 'high' when more than 10% out of range", () => {
    // wbc refLow 4.0 — 3.0 is 25% below
    expect(classifyLabValue("wbc", 3.0)).toBe("low");
    // platelets refHigh 400 — 500 is 25% above
    expect(classifyLabValue("platelets", 500)).toBe("high");
  });

  it("escalates to critical for values below criticalLow / above criticalHigh", () => {
    expect(classifyLabValue("platelets", 30)).toBe("critical_low");
    expect(classifyLabValue("wbc", 35)).toBe("critical_high");
    // ANC criticalLow is 0.5 — neutropenic zone must trip critical_low
    expect(classifyLabValue("anc", 0.4)).toBe("critical_low");
  });
});

describe("labStatusTone", () => {
  it("maps critical statuses to 'danger'", () => {
    expect(labStatusTone("critical_low")).toBe("danger");
    expect(labStatusTone("critical_high")).toBe("danger");
  });
  it("maps out-of-range non-critical to 'warning'", () => {
    expect(labStatusTone("low")).toBe("warning");
    expect(labStatusTone("high")).toBe("warning");
    expect(labStatusTone("borderline")).toBe("warning");
  });
  it("maps in_range to 'success' and unknown to 'neutral'", () => {
    expect(labStatusTone("in_range")).toBe("success");
    expect(labStatusTone("unknown")).toBe("neutral");
  });
});

describe("labStatusLabel", () => {
  it("returns patient-safe copy for every status", () => {
    expect(labStatusLabel("critical_low")).toBe("Very low");
    expect(labStatusLabel("critical_high")).toBe("Very high");
    expect(labStatusLabel("in_range")).toBe("In range");
    expect(labStatusLabel("borderline")).toBe("Borderline");
    expect(labStatusLabel("unknown")).toBe("No value");
  });
});

describe("severityBucket", () => {
  it("buckets 0-3 as mild, 4-6 as moderate, 7+ as severe", () => {
    expect(severityBucket(0)).toBe("mild");
    expect(severityBucket(3)).toBe("mild");
    expect(severityBucket(4)).toBe("moderate");
    expect(severityBucket(6)).toBe("moderate");
    expect(severityBucket(7)).toBe("severe");
    expect(severityBucket(10)).toBe("severe");
  });
});

describe("LAB_REFERENCE_RANGES safety invariants", () => {
  // These invariants protect against accidental config drift — if a future
  // edit inverts refLow/refHigh or removes the disclaimer, the test fails.
  it.each(Object.entries(LAB_REFERENCE_RANGES))(
    "%s has well-ordered thresholds and required copy",
    (_key, range) => {
      expect(range.criticalLow).toBeLessThan(range.refLow);
      expect(range.refLow).toBeLessThan(range.refHigh);
      expect(range.refHigh).toBeLessThan(range.criticalHigh);
      expect(range.label.length).toBeGreaterThan(0);
      expect(range.unit.length).toBeGreaterThan(0);
      expect(range.description.length).toBeGreaterThan(20);
      expect(range.disclaimer.length).toBeGreaterThan(20);
    },
  );
});

describe("NON_DIAGNOSTIC_DISCLAIMER", () => {
  it("explicitly states the portal does not diagnose or replace clinician judgement", () => {
    expect(NON_DIAGNOSTIC_DISCLAIMER.toLowerCase()).toContain("diagnose");
    expect(NON_DIAGNOSTIC_DISCLAIMER.toLowerCase()).toContain("clinician");
  });
});
