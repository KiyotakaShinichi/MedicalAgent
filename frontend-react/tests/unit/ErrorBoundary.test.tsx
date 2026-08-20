import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { ErrorBoundary } from "../../src/components/ui/ErrorBoundary";
import {
  registerTelemetrySink,
  resetTelemetrySinks,
  type TelemetryEvent,
} from "../../src/lib/telemetry";

function Explode({ message = "render exploded" }: { message?: string }): never {
  throw new Error(message);
}

describe("ErrorBoundary", () => {
  beforeEach(() => {
    resetTelemetrySinks();
    // React logs the caught error itself; silence it so the run stays readable.
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    resetTelemetrySinks();
    vi.restoreAllMocks();
  });

  it("renders children when nothing throws", () => {
    render(
      <ErrorBoundary surface="the test surface">
        <p>healthy content</p>
      </ErrorBoundary>,
    );
    expect(screen.getByText("healthy content")).toBeInTheDocument();
  });

  it("shows a recoverable alert instead of a blank screen", () => {
    render(
      <ErrorBoundary surface="the safety section">
        <Explode />
      </ErrorBoundary>,
    );

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(/Something went wrong rendering the safety section/i);
    expect(alert).toHaveTextContent("render exploded");
    expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument();
  });

  it("reports the crash to telemetry as an unexpected failure", () => {
    const events: TelemetryEvent[] = [];
    registerTelemetrySink((e) => events.push(e));

    render(
      <ErrorBoundary surface="the admin dashboard">
        <Explode message="boom" />
      </ErrorBoundary>,
    );

    expect(events).toHaveLength(1);
    expect(events[0].kind).toBe("unexpected");
    expect(events[0].surface).toBe("ErrorBoundary:the admin dashboard");
    expect(events[0].message).toBe("boom");
  });

  it("recovers when the user retries and the child no longer throws", async () => {
    const user = userEvent.setup();

    function Flaky() {
      const [ok, setOk] = useState(false);
      return (
        <>
          <button type="button" onClick={() => setOk(true)}>fix it</button>
          <ErrorBoundary surface="the panel">{ok ? <p>recovered</p> : <Explode />}</ErrorBoundary>
        </>
      );
    }

    render(<Flaky />);
    expect(screen.getByRole("alert")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "fix it" }));
    await user.click(screen.getByRole("button", { name: /try again/i }));

    expect(screen.getByText("recovered")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("does not let a failing telemetry sink mask the fallback UI", () => {
    registerTelemetrySink(() => {
      throw new Error("monitoring is down");
    });

    render(
      <ErrorBoundary surface="the panel">
        <Explode />
      </ErrorBoundary>,
    );

    expect(screen.getByRole("alert")).toBeInTheDocument();
  });
});
