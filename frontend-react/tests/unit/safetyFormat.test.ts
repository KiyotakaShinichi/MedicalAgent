import { describe, it, expect } from "vitest";
import {
  statusBadge,
  fmtRate,
  fmtScore,
  highIsGood,
  lowIsGood,
  coerceIntegrityStatus,
  readRecord,
  readString,
  readNumber,
  readArray,
} from "../../src/pages/admin/sections/safety/safetyFormat";

describe("statusBadge", () => {
  it("maps known passing states to green", () => {
    for (const s of ["passed", "strong", "available", "ok"]) {
      expect(statusBadge(s)).toBe("green");
    }
  });

  it("maps partial states to amber and failures to red", () => {
    expect(statusBadge("needs_attention")).toBe("amber");
    expect(statusBadge("partial")).toBe("amber");
    expect(statusBadge("failed")).toBe("red");
    expect(statusBadge("error")).toBe("red");
  });

  it("never treats an unknown or absent status as a pass", () => {
    // This is the safety-relevant property: a status the frontend does not
    // recognise must not render green on an evaluation dashboard.
    expect(statusBadge(undefined)).toBe("muted");
    expect(statusBadge("")).toBe("muted");
    expect(statusBadge("some_new_backend_state")).toBe("muted");
  });
});

describe("fmtRate / fmtScore", () => {
  it("renders an em dash for absent values rather than 0", () => {
    expect(fmtRate(null)).toBe("—");
    expect(fmtRate(undefined)).toBe("—");
    expect(fmtScore(null)).toBe("—");
    expect(fmtScore(undefined)).toBe("—");
  });

  it("formats rates as whole percentages", () => {
    expect(fmtRate(0)).toBe("0%");
    expect(fmtRate(0.925)).toBe("93%");
    expect(fmtRate(1)).toBe("100%");
  });

  it("formats scores at the requested precision", () => {
    expect(fmtScore(0.123456)).toBe("0.123");
    expect(fmtScore(0.123456, 2)).toBe("0.12");
  });
});

describe("threshold tones", () => {
  it("grades high-is-good metrics across the three bands", () => {
    expect(highIsGood(0.95)).toBe("green");
    expect(highIsGood(0.85)).toBe("amber");
    expect(highIsGood(0.5)).toBe("red");
  });

  it("grades low-is-good metrics across the three bands", () => {
    expect(lowIsGood(0)).toBe("green");
    expect(lowIsGood(0.01)).toBe("amber");
    expect(lowIsGood(0.5)).toBe("red");
  });

  it("treats an unmeasured metric as muted, not as passing", () => {
    expect(highIsGood(null)).toBe("muted");
    expect(highIsGood(undefined)).toBe("muted");
    expect(lowIsGood(null)).toBe("muted");
    expect(lowIsGood(undefined)).toBe("muted");
  });
});

describe("coerceIntegrityStatus", () => {
  it("passes through the known integrity states", () => {
    expect(coerceIntegrityStatus("passed")).toBe("passed");
    expect(coerceIntegrityStatus("failed")).toBe("failed");
    expect(coerceIntegrityStatus("skipped")).toBe("skipped");
  });

  it("falls back to unknown for anything unrecognised", () => {
    expect(coerceIntegrityStatus("weird")).toBe("unknown");
    expect(coerceIntegrityStatus(undefined)).toBe("unknown");
  });
});

describe("artifact readers", () => {
  const artifact = {
    metrics: { pass_rate: 0.9, bad: Number.NaN, label: "ok" },
    failures: [1, 2, 3],
    scalar: 5,
    listy: ["a"],
  };

  it("returns nested records only for plain objects", () => {
    expect(readRecord(artifact, "metrics")).toEqual(artifact.metrics);
    // An array is not a record — returning it would break `.foo` reads downstream.
    expect(readRecord(artifact, "listy")).toBeUndefined();
    expect(readRecord(artifact, "scalar")).toBeUndefined();
    expect(readRecord(undefined, "metrics")).toBeUndefined();
  });

  it("returns typed primitives and undefined on mismatch", () => {
    const metrics = readRecord(artifact, "metrics");
    expect(readNumber(metrics, "pass_rate")).toBe(0.9);
    expect(readString(metrics, "label")).toBe("ok");
    expect(readString(metrics, "pass_rate")).toBeUndefined();
    expect(readNumber(metrics, "label")).toBeUndefined();
  });

  it("treats NaN as absent so it renders as an em dash, not NaN%", () => {
    const metrics = readRecord(artifact, "metrics");
    expect(readNumber(metrics, "bad")).toBeUndefined();
    expect(fmtRate(readNumber(metrics, "bad"))).toBe("—");
  });

  it("returns an empty array for missing or non-array values", () => {
    expect(readArray(artifact, "failures")).toEqual([1, 2, 3]);
    expect(readArray(artifact, "scalar")).toEqual([]);
    expect(readArray(artifact, "nope")).toEqual([]);
  });
});
