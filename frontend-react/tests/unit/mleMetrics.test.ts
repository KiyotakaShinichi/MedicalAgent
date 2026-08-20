import { describe, it, expect } from "vitest";
import {
  parseMetricValue,
  gradeMetric,
  displayMetricValue,
  buildGlossaryValues,
  toRecord,
  HOLDOUT_METRICS,
  TRAINING_REPORT_METRICS,
  type MetricSpec,
} from "../../src/pages/admin/sections/mle/mleMetrics";

const HIGHER: MetricSpec = { label: "AUROC", direction: "higher-is-better", threshold: 0.85, amberFactor: 0.85 };
const LOWER: MetricSpec = { label: "Brier", direction: "lower-is-better", threshold: 0.1, amberFactor: 2 };
const UNGRADED: MetricSpec = { label: "Best classifier", direction: "higher-is-better", threshold: null, amberFactor: 1 };

describe("parseMetricValue", () => {
  it("accepts numbers and numeric strings", () => {
    expect(parseMetricValue(0.91)).toBe(0.91);
    expect(parseMetricValue("0.91")).toBe(0.91);
    expect(parseMetricValue(" 0.91 ")).toBe(0.91);
  });

  it("preserves a genuine zero", () => {
    // Regression: the previous inline code used `parseFloat(x) || null`, which
    // mapped a measured 0 to null because 0 is falsy — a real zero was shown
    // as "not measured".
    expect(parseMetricValue(0)).toBe(0);
    expect(parseMetricValue("0")).toBe(0);
    expect(parseMetricValue("0.0")).toBe(0);
  });

  it("rejects non-numeric and non-finite input rather than yielding NaN", () => {
    expect(parseMetricValue("gradient_boosting")).toBeNull();
    expect(parseMetricValue(null)).toBeNull();
    expect(parseMetricValue(undefined)).toBeNull();
    expect(parseMetricValue("")).toBeNull();
    expect(parseMetricValue("   ")).toBeNull();
    expect(parseMetricValue(Number.NaN)).toBeNull();
    expect(parseMetricValue(Number.POSITIVE_INFINITY)).toBeNull();
    expect(parseMetricValue({})).toBeNull();
  });
});

describe("gradeMetric", () => {
  it("grades higher-is-better metrics across the bands", () => {
    expect(gradeMetric(0.9, HIGHER)).toBe("green");
    expect(gradeMetric(0.85, HIGHER)).toBe("green");
    // 0.85 * 0.85 = 0.7225 — amber floor.
    expect(gradeMetric(0.75, HIGHER)).toBe("amber");
    expect(gradeMetric(0.5, HIGHER)).toBe("red");
  });

  it("grades lower-is-better metrics across the bands", () => {
    expect(gradeMetric(0.05, LOWER)).toBe("green");
    expect(gradeMetric(0.1, LOWER)).toBe("green");
    expect(gradeMetric(0.15, LOWER)).toBe("amber");
    expect(gradeMetric(0.5, LOWER)).toBe("red");
  });

  it("mutes an unparseable or absent value instead of grading it green", () => {
    // The safety rule: "could not measure" must never look like "passed".
    expect(gradeMetric(null, HIGHER)).toBe("muted");
    expect(gradeMetric(undefined, LOWER)).toBe("muted");
    expect(gradeMetric("n/a", HIGHER)).toBe("muted");
    expect(gradeMetric(Number.NaN, LOWER)).toBe("muted");
  });

  it("mutes descriptive fields that carry no threshold", () => {
    expect(gradeMetric("gradient_boosting", UNGRADED)).toBe("muted");
    expect(gradeMetric(42, UNGRADED)).toBe("muted");
  });

  it("does not invert direction when a label is renamed", () => {
    // Regression: grading direction used to be chosen by comparing the visible
    // label against string literals, so renaming a label silently flipped it.
    const renamed: MetricSpec = { ...LOWER, label: "Calibration error" };
    expect(gradeMetric(0.05, renamed)).toBe("green");
    expect(gradeMetric(0.5, renamed)).toBe("red");
  });
});

describe("metric spec tables", () => {
  it("marks descriptive training-report fields as ungraded", () => {
    const ungraded = TRAINING_REPORT_METRICS.filter((s) => s.threshold === null).map((s) => s.label);
    expect(ungraded).toEqual(["Test patients", "Best classifier", "Best regressor"]);
  });

  it("keeps every holdout metric graded and directional", () => {
    for (const spec of HOLDOUT_METRICS) {
      expect(spec.threshold).not.toBeNull();
      expect(["higher-is-better", "lower-is-better"]).toContain(spec.direction);
    }
  });

  it("orients error-style metrics as lower-is-better", () => {
    const lower = HOLDOUT_METRICS.filter((s) => s.direction === "lower-is-better").map((s) => s.label);
    expect(lower).toEqual(["Brier", "MAE"]);
  });
});

describe("displayMetricValue", () => {
  it("returns null for absent values so the card renders its own dash", () => {
    expect(displayMetricValue(null)).toBeNull();
    expect(displayMetricValue(undefined)).toBeNull();
  });

  it("stringifies present values, including zero", () => {
    expect(displayMetricValue(0)).toBe("0");
    expect(displayMetricValue(0.912)).toBe("0.912");
    expect(displayMetricValue("logreg")).toBe("logreg");
  });
});

describe("toRecord", () => {
  it("passes through plain objects", () => {
    expect(toRecord({ auroc: 1 })).toEqual({ auroc: 1 });
  });

  it("rejects arrays, primitives, and nullish values", () => {
    expect(toRecord([1, 2])).toBeUndefined();
    expect(toRecord("nope")).toBeUndefined();
    expect(toRecord(5)).toBeUndefined();
    expect(toRecord(null)).toBeUndefined();
    expect(toRecord(undefined)).toBeUndefined();
  });
});

describe("buildGlossaryValues", () => {
  it("returns an empty map when there is no result", () => {
    expect(buildGlossaryValues(undefined)).toEqual({});
  });

  it("maps the five glossary metrics and nulls what is missing", () => {
    expect(buildGlossaryValues({ auroc: 0.93, brier_score: "0.08", ece: 0 })).toEqual({
      AUROC: 0.93,
      "Brier Score": 0.08,
      // A measured zero ECE is perfect calibration, not missing data.
      ECE: 0,
      "Sensitivity (Recall)": null,
      "MAE (Regression)": null,
    });
  });

  it("nulls non-numeric values rather than propagating NaN", () => {
    const values = buildGlossaryValues({ auroc: "unavailable" });
    expect(values.AUROC).toBeNull();
    expect(Number.isNaN(values.AUROC as number)).toBe(false);
  });
});
