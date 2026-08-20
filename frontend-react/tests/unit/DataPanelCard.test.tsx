import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DataPanelCard } from "../../src/pages/admin/sections/mle/DataPanelCard";
import { PanelErrorNotice } from "../../src/pages/admin/sections/mle/PanelErrorNotice";
import { expectNoA11yViolations } from "../a11y";

const BASE = {
  title: "Leakage Audit",
  loading: false,
  error: null,
  empty: false,
  emptyLabel: "No leakage audit available",
  errorLabel: "Could not load leakage audit",
};

describe("DataPanelCard", () => {
  it("renders children when loaded with data", () => {
    render(<DataPanelCard {...BASE}><p>audit body</p></DataPanelCard>);
    expect(screen.getByText("audit body")).toBeInTheDocument();
    expect(screen.getByText("Leakage Audit")).toBeInTheDocument();
  });

  it("shows loading ahead of everything else", () => {
    render(<DataPanelCard {...BASE} loading empty error="boom"><p>audit body</p></DataPanelCard>);
    expect(screen.queryByText("audit body")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("prefers the error state over the empty state", () => {
    // Regression: panels used to check `empty` before `error`, so a failed
    // fetch fell through to "no artifact exists" — telling the operator the
    // evidence was absent when the request had actually failed.
    render(
      <DataPanelCard {...BASE} error="connection refused" empty>
        <p>audit body</p>
      </DataPanelCard>,
    );

    expect(screen.getByText(/connection refused/)).toBeInTheDocument();
    expect(screen.queryByText("No leakage audit available")).not.toBeInTheDocument();
  });

  it("includes the panel-specific label in the error message", () => {
    render(<DataPanelCard {...BASE} error="503"><p>body</p></DataPanelCard>);
    expect(screen.getByText(/Could not load leakage audit: 503/)).toBeInTheDocument();
  });

  it("shows the empty state when there is no artifact and no error", () => {
    render(<DataPanelCard {...BASE} empty><p>audit body</p></DataPanelCard>);
    expect(screen.getByText("No leakage audit available")).toBeInTheDocument();
    expect(screen.queryByText("audit body")).not.toBeInTheDocument();
  });

  it("renders an optional provenance tag", () => {
    render(
      <DataPanelCard {...BASE} tag={{ label: "Synthetic data", background: "#000", color: "#fff" }}>
        <p>body</p>
      </DataPanelCard>,
    );
    expect(screen.getByText("Synthetic data")).toBeInTheDocument();
  });

  it("gives the action button a name that identifies its panel", async () => {
    // "Refresh" appears on many panels; the accessible name must disambiguate.
    const onClick = vi.fn();
    const user = userEvent.setup();
    render(
      <DataPanelCard {...BASE} action={{ label: "Rerun audit", onClick, running: false }}>
        <p>body</p>
      </DataPanelCard>,
    );

    const button = screen.getByRole("button", { name: /Rerun audit — Leakage Audit/ });
    await user.click(button);
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("marks the action busy and disabled while running", () => {
    render(
      <DataPanelCard {...BASE} action={{ label: "Rerun audit", onClick: () => {}, running: true }}>
        <p>body</p>
      </DataPanelCard>,
    );
    const button = screen.getByRole("button", { name: /Rerun audit/ });
    expect(button).toHaveAttribute("aria-busy", "true");
    expect(button).toBeDisabled();
  });

  it("keeps the action usable while the panel shows an error, so a run can be retried", () => {
    render(
      <DataPanelCard {...BASE} error="timeout" action={{ label: "Rerun audit", onClick: () => {}, running: false }}>
        <p>body</p>
      </DataPanelCard>,
    );
    expect(screen.getByRole("button", { name: /Rerun audit/ })).toBeEnabled();
  });

  it("has no detectable accessibility violations", async () => {
    const { container } = render(
      <DataPanelCard
        {...BASE}
        tag={{ label: "Synthetic data", background: "#000", color: "#fff" }}
        action={{ label: "Rerun audit", onClick: () => {}, running: false }}
      >
        <p>audit body</p>
      </DataPanelCard>,
    );
    await expectNoA11yViolations(container);
  });
});

describe("PanelErrorNotice", () => {
  it("renders nothing when there is no error", () => {
    const { container } = render(<PanelErrorNotice panel="Leakage audit" error={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("names the failing panel and warns that shown values may be stale", () => {
    render(<PanelErrorNotice panel="Leakage audit" error="job crashed" />);

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Leakage audit could not be updated.");
    expect(alert).toHaveTextContent("job crashed");
    // The operator must not read the still-rendered artifact as current.
    expect(alert).toHaveTextContent(/previous run and may be out of date/i);
  });
});
