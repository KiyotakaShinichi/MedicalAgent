import { describe, it, expect } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { bytesToBase64, readFileAsBase64 } from "../../src/lib/fileEncoding";
import { touchesPatientReport, REPORT_TOUCHING_ACTION_TYPES } from "../../src/pages/patient/savedActions";
import { MedLogPanel } from "../../src/pages/patient/MedLogPanel";
import type { MedicationLog, SavedAction } from "../../src/types/api";
import { expectNoA11yViolations } from "../a11y";

describe("fileEncoding", () => {
  it("encodes bytes as base64", () => {
    expect(bytesToBase64(new Uint8Array([72, 101, 108, 108, 111]))).toBe("SGVsbG8=");
  });

  it("encodes an empty buffer", () => {
    expect(bytesToBase64(new Uint8Array([]))).toBe("");
  });

  it("encodes payloads larger than the chunk window", () => {
    // Regression guard: a single `fromCharCode(...bytes)` over a large file
    // overflows the argument limit and throws, which is why the encoder walks
    // the array in fixed windows. 0x8000 is the window, so cross it.
    const size = 0x8000 * 2 + 123;
    const bytes = new Uint8Array(size).fill(65);
    const encoded = bytesToBase64(bytes);
    expect(encoded).toBe(btoa("A".repeat(size)));
  });

  it("preserves high bytes that are not valid UTF-8 text", () => {
    expect(bytesToBase64(new Uint8Array([0x00, 0xff, 0x80]))).toBe(btoa("\x00\xff\x80"));
  });

  it("reads a Blob and returns a bare base64 payload with no data-URL prefix", async () => {
    const encoded = await readFileAsBase64(new Blob([new Uint8Array([72, 105])]));
    expect(encoded).toBe("SGk=");
    expect(encoded).not.toContain("base64,");
    expect(encoded).not.toContain("data:");
  });
});

describe("touchesPatientReport", () => {
  const action = (type: string) => ({ type } as SavedAction);

  it("is true for a record write, so the dashboard refetches", () => {
    expect(touchesPatientReport([action("saved_labs")])).toBe(true);
    expect(touchesPatientReport([action("saved_symptom")])).toBe(true);
  });

  it("accepts both the legacy and current action names", () => {
    // The backend has emitted both spellings; missing either would leave the
    // patient looking at pre-save data after a successful write.
    expect(touchesPatientReport([action("save_lab")])).toBe(true);
    expect(touchesPatientReport([action("saved_labs")])).toBe(true);
    expect(touchesPatientReport([action("save_mri")])).toBe(true);
  });

  it("is false for actions that do not change the record", () => {
    expect(touchesPatientReport([action("answered_question")])).toBe(false);
    expect(touchesPatientReport([action("possible_metastatic_indicator")])).toBe(false);
  });

  it("is false for an empty batch", () => {
    expect(touchesPatientReport([])).toBe(false);
  });

  it("is true when any action in a mixed batch writes to the record", () => {
    expect(touchesPatientReport([action("answered_question"), action("saved_medication")])).toBe(true);
  });

  it("treats an unrecognised action as non-touching", () => {
    // Fail-closed on refetching is the safe default; a new backend write
    // action must be registered in the set explicitly.
    expect(touchesPatientReport([action("brand_new_backend_action")])).toBe(false);
  });

  it("covers all four record domains", () => {
    for (const domain of ["symptom", "lab", "medication", "imaging"]) {
      const matches = [...REPORT_TOUCHING_ACTION_TYPES].filter((t) => t.includes(domain));
      expect(matches.length, `no action type covers ${domain}`).toBeGreaterThan(0);
    }
  });
});

describe("MedLogPanel", () => {
  const med = (overrides: Partial<MedicationLog> = {}): MedicationLog => ({
    date: "2026-03-01",
    medication: "Doxorubicin",
    dose: "60mg/m2",
    frequency: "q21d",
    ...overrides,
  }) as MedicationLog;

  it("names who can add entries when the log is empty", () => {
    render(<MedLogPanel meds={[]} />);
    expect(screen.getByText(/your care team can add these/i)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("orders entries newest first", () => {
    render(
      <MedLogPanel
        meds={[
          med({ date: "2026-01-01", medication: "Older" }),
          med({ date: "2026-06-01", medication: "Newer" }),
        ]}
      />,
    );

    const rows = screen.getAllByRole("row").slice(1); // drop the header row
    expect(within(rows[0]).getByText("Newer")).toBeInTheDocument();
    expect(within(rows[1]).getByText("Older")).toBeInTheDocument();
  });

  it("does not reorder the caller's array in place", () => {
    // The same report object feeds several panels; sorting the prop would
    // reorder medications everywhere else on the dashboard.
    const meds = [med({ date: "2026-01-01", medication: "Older" }), med({ date: "2026-06-01", medication: "Newer" })];
    render(<MedLogPanel meds={meds} />);
    expect(meds[0].medication).toBe("Older");
  });

  it("caps the table and says how many more exist", () => {
    const meds = Array.from({ length: 15 }, (_, i) =>
      med({ date: `2026-01-${String(i + 1).padStart(2, "0")}`, medication: `Med ${i}` }),
    );
    render(<MedLogPanel meds={meds} />);

    expect(screen.getAllByRole("row").slice(1)).toHaveLength(12);
    expect(screen.getByText("+ 3 more in patient record")).toBeInTheDocument();
    expect(screen.getByText("15 entries")).toBeInTheDocument();
  });

  it("truncates a timestamp to its date portion", () => {
    render(<MedLogPanel meds={[med({ date: "2026-03-01T14:22:00Z" })]} />);
    expect(screen.getByText("2026-03-01")).toBeInTheDocument();
  });

  it("renders rows whose date is missing without crashing", () => {
    const malformed = { medication: "Unknown start", dose: "-", frequency: "-" } as unknown as MedicationLog;
    render(<MedLogPanel meds={[malformed]} />);
    expect(screen.getByText("Unknown start")).toBeInTheDocument();
  });

  it("has no detectable accessibility violations", async () => {
    const { container } = render(<MedLogPanel meds={[med(), med({ date: "2026-02-01" })]} />);
    await expectNoA11yViolations(container);
  });
});
