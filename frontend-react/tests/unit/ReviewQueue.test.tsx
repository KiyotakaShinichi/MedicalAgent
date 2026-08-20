import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReviewQueue } from "../../src/pages/clinician/ReviewQueue";
import type { ReviewQueueItem } from "../../src/types/api";
import { expectNoA11yViolations } from "../a11y";

function item(overrides: Partial<ReviewQueueItem> = {}): ReviewQueueItem {
  return {
    patient_id: "P001",
    patient_name: "Ana Reyes",
    overall_status: "needs_review",
    priority_score: 82,
    urgent_flags: [],
    latest_decision: null,
    ...overrides,
  };
}

describe("ReviewQueue", () => {
  it("shows an empty state when no patients are queued", () => {
    render(<ReviewQueue queue={[]} selectedId={null} onSelect={() => {}} />);
    expect(screen.getByText("Queue empty")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("renders one selectable entry per patient with id and status", () => {
    render(
      <ReviewQueue
        queue={[item(), item({ patient_id: "P002", patient_name: "Bea Cruz", overall_status: "stable" })]}
        selectedId={null}
        onSelect={() => {}}
      />,
    );

    const entries = screen.getAllByRole("listitem");
    expect(entries).toHaveLength(2);
    expect(within(entries[0]).getByText("Ana Reyes")).toBeInTheDocument();
    expect(within(entries[0]).getByText("P001")).toBeInTheDocument();
    // Underscores are humanised for display.
    expect(within(entries[0]).getByText("needs review")).toBeInTheDocument();
    expect(within(entries[1]).getByText("stable")).toBeInTheDocument();
  });

  it("preserves the backend's triage order rather than re-sorting", () => {
    // Clinical prioritisation is the backend's decision. If this component
    // ever re-ranked, a lower-priority patient could be surfaced first.
    render(
      <ReviewQueue
        queue={[
          item({ patient_id: "P001", patient_name: "Low", priority_score: 10 }),
          item({ patient_id: "P002", patient_name: "High", priority_score: 99 }),
        ]}
        selectedId={null}
        onSelect={() => {}}
      />,
    );

    const names = screen.getAllByRole("listitem").map((li) => within(li).getByRole("button").textContent);
    expect(names[0]).toContain("Low");
    expect(names[1]).toContain("High");
  });

  it("calls onSelect with the patient id when an entry is activated", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(<ReviewQueue queue={[item(), item({ patient_id: "P002", patient_name: "Bea Cruz" })]} selectedId={null} onSelect={onSelect} />);

    await user.click(screen.getByRole("button", { name: /Bea Cruz/ }));

    expect(onSelect).toHaveBeenCalledExactlyOnceWith("P002");
  });

  it("is operable by keyboard", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(<ReviewQueue queue={[item()]} selectedId={null} onSelect={onSelect} />);

    await user.tab();
    expect(screen.getByRole("button", { name: /Ana Reyes/ })).toHaveFocus();
    await user.keyboard("{Enter}");

    expect(onSelect).toHaveBeenCalledWith("P001");
  });

  it("marks only the selected entry with aria-current", () => {
    render(
      <ReviewQueue
        queue={[item(), item({ patient_id: "P002", patient_name: "Bea Cruz" })]}
        selectedId="P002"
        onSelect={() => {}}
      />,
    );

    expect(screen.getByRole("button", { name: /Bea Cruz/ })).toHaveAttribute("aria-current", "true");
    expect(screen.getByRole("button", { name: /Ana Reyes/ })).not.toHaveAttribute("aria-current");
  });

  it("surfaces urgent flags with a count", () => {
    render(
      <ReviewQueue
        queue={[item({ urgent_flags: ["neutropenia", "fever"] })]}
        selectedId={null}
        onSelect={() => {}}
      />,
    );
    expect(screen.getByText("2 urgent")).toBeInTheDocument();
  });

  it("omits the urgent indicator when there are no flags", () => {
    render(<ReviewQueue queue={[item({ urgent_flags: [] })]} selectedId={null} onSelect={() => {}} />);
    expect(screen.queryByText(/urgent/)).not.toBeInTheDocument();
  });

  it("rounds the priority score for display", () => {
    render(<ReviewQueue queue={[item({ priority_score: 82.6 })]} selectedId={null} onSelect={() => {}} />);
    expect(screen.getByText("Priority 83")).toBeInTheDocument();
  });

  it("shows the last review decision only when one exists", () => {
    const { rerender } = render(
      <ReviewQueue queue={[item({ latest_decision: "approved" })]} selectedId={null} onSelect={() => {}} />,
    );
    expect(screen.getByText("Last review: approved")).toBeInTheDocument();

    rerender(<ReviewQueue queue={[item({ latest_decision: null })]} selectedId={null} onSelect={() => {}} />);
    expect(screen.queryByText(/Last review/)).not.toBeInTheDocument();
  });

  describe("malformed API responses", () => {
    it("renders an entry missing its status, priority, and flags without crashing", () => {
      // The queue endpoint has returned partial rows; a missing field must not
      // blank the whole clinician workspace.
      const malformed = {
        patient_id: "P009",
        patient_name: "Partial Record",
      } as unknown as ReviewQueueItem;

      render(<ReviewQueue queue={[malformed]} selectedId={null} onSelect={() => {}} />);

      expect(screen.getByText("Partial Record")).toBeInTheDocument();
      // Neutral fallback status, not a resolved-looking one.
      expect(screen.getByText("review")).toBeInTheDocument();
      // An unscored patient shows a dash, never "Priority 0", which would read
      // as the lowest priority rather than "not scored".
      expect(screen.getByText("Priority -")).toBeInTheDocument();
      expect(screen.queryByText(/urgent/)).not.toBeInTheDocument();
    });

    it("does not treat a zero priority as missing", () => {
      render(<ReviewQueue queue={[item({ priority_score: 0 })]} selectedId={null} onSelect={() => {}} />);
      expect(screen.getByText("Priority 0")).toBeInTheDocument();
    });
  });

  it("has no detectable accessibility violations", async () => {
    const { container } = render(
      <ReviewQueue
        queue={[item({ urgent_flags: ["fever"], latest_decision: "edited" }), item({ patient_id: "P002", patient_name: "Bea Cruz" })]}
        selectedId="P001"
        onSelect={() => {}}
      />,
    );
    await expectNoA11yViolations(container);
  });
});
