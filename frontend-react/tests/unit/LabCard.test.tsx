import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { LabCard } from "../../src/components/ui/LabCard";

describe("LabCard", () => {
  it("renders an em-dash and 'No value' chip when value is null", () => {
    render(<LabCard labKey="wbc" value={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.getByText("No value")).toBeInTheDocument();
  });

  it("shows the In range chip for a value inside the reference band", () => {
    // wbc band is 4.0–11.0, so 7.0 should be in_range.
    render(<LabCard labKey="wbc" value={7.0} />);
    expect(screen.getByText("In range")).toBeInTheDocument();
    expect(screen.getByText("7.0")).toBeInTheDocument();
  });

  it("flags 'Very low' for a critical platelet count", () => {
    // platelets criticalLow is 50.0 — 30 must surface critical_low ("Very low").
    render(<LabCard labKey="platelets" value={30} />);
    expect(screen.getByText("Very low")).toBeInTheDocument();
    expect(screen.getByText("30")).toBeInTheDocument(); // platelets render as integer
  });

  it("flags 'Very high' for a critical WBC value", () => {
    render(<LabCard labKey="wbc" value={35} />);
    expect(screen.getByText("Very high")).toBeInTheDocument();
  });

  it("renders a sparkline path when history has >= 2 points", () => {
    const { container } = render(
      <LabCard
        labKey="hemoglobin"
        value={13.5}
        history={[
          { date: "2026-04-01", value: 12.8 },
          { date: "2026-04-15", value: 13.1 },
          { date: "2026-05-01", value: 13.5 },
        ]}
      />,
    );
    // sparkline svg is labelled by role="img"
    expect(container.querySelector('svg[aria-label="Sparkline of recent values"]')).not.toBeNull();
  });

  it("renders the reference range in the footer using lab constants", () => {
    render(<LabCard labKey="platelets" value={250} />);
    // platelets refLow=150, refHigh=400, unit="K/uL"
    expect(screen.getAllByText(/150–400/)[0]).toBeInTheDocument();
  });
});
