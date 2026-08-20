import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  redactString,
  redactUrlPath,
  redactValue,
  registerTelemetrySink,
  reportError,
  reportExpectedError,
  resetTelemetrySinks,
  installGlobalErrorHandlers,
  type TelemetryEvent,
} from "../../src/lib/telemetry";

describe("telemetry redaction", () => {
  it("strips JWT-shaped bearer tokens from messages", () => {
    const jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJQMDAxIn0.abcdefghijklmnop";
    const out = redactString(`Request failed with token ${jwt}`);
    expect(out).not.toContain(jwt);
    expect(out).toContain("[redacted-token]");
  });

  it("strips Authorization-style bearer prefixes", () => {
    const out = redactString("Bearer abcdef1234567890abcdef rejected");
    expect(out).not.toContain("abcdef1234567890abcdef");
    expect(out).toContain("[redacted-token]");
  });

  it("strips email addresses", () => {
    const out = redactString("failed for patient.name@hospital.org");
    expect(out).not.toContain("patient.name@hospital.org");
    expect(out).toContain("[redacted-email]");
  });

  it("truncates long free text that could carry clinical prose", () => {
    const prose = "a".repeat(500);
    const out = redactString(prose);
    expect(out.length).toBeLessThan(prose.length);
    expect(out).toContain("[truncated]");
  });

  it("redacts values under sensitive keys but keeps benign structure", () => {
    const out = redactValue({
      status: 500,
      access_token: "super-secret",
      patient_id: "P001",
      notes: "patient reports severe pain",
      route: "/safety/center",
    }) as Record<string, unknown>;

    expect(out.status).toBe(500);
    expect(out.route).toBe("/safety/center");
    expect(out.access_token).toBe("[redacted]");
    expect(out.patient_id).toBe("[redacted]");
    expect(out.notes).toBe("[redacted]");
  });

  it("bounds recursion instead of following deeply nested payloads forever", () => {
    // Depth 6 exceeds the depth-4 budget, so the innermost value is dropped.
    const deep = { a: { b: { c: { d: { e: { f: "leaf" } } } } } };
    expect(JSON.stringify(redactValue(deep))).toContain("[redacted-depth]");
  });

  it("templates identifiers out of request paths", () => {
    expect(redactUrlPath("/patients/P001/labs")).toBe("/patients/:id/labs");
    expect(redactUrlPath("/reviews/42")).toBe("/reviews/:id");
    expect(redactUrlPath("/jobs/6f1a2b3c-4d5e-6f70-8192-a3b4c5d6e7f8")).toBe("/jobs/:id");
  });

  it("drops query strings entirely rather than guessing which params are safe", () => {
    expect(redactUrlPath("/search?q=lump+in+breast")).toBe("/search?[redacted-query]");
  });

  it("leaves plain route segments untouched so routes stay groupable", () => {
    expect(redactUrlPath("/safety/center")).toBe("/safety/center");
  });
});

describe("reportError", () => {
  beforeEach(() => {
    resetTelemetrySinks();
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    resetTelemetrySinks();
    vi.restoreAllMocks();
  });

  it("defaults to the unexpected kind so unclassified failures are not hidden", () => {
    const event = reportError(new Error("boom"), { surface: "test" });
    expect(event.kind).toBe("unexpected");
    expect(console.error).toHaveBeenCalled();
  });

  it("logs expected failures at warn, not error", () => {
    reportExpectedError(new Error("401 unauthorised"), "api.login");
    expect(console.warn).toHaveBeenCalled();
    expect(console.error).not.toHaveBeenCalled();
  });

  it("fans out to registered sinks with a redacted payload", () => {
    const events: TelemetryEvent[] = [];
    registerTelemetrySink((e) => events.push(e));

    reportError(new Error("failed"), {
      surface: "api.request",
      detail: { authorization: "Bearer sekrit", status: 500 },
    });

    expect(events).toHaveLength(1);
    expect(events[0].surface).toBe("api.request");
    expect(events[0].detail.authorization).toBe("[redacted]");
    expect(events[0].detail.status).toBe(500);
  });

  it("does not let a throwing sink escape to the caller", () => {
    registerTelemetrySink(() => {
      throw new Error("monitoring provider is down");
    });
    const good: TelemetryEvent[] = [];
    registerTelemetrySink((e) => good.push(e));

    expect(() => reportError(new Error("app failure"), { surface: "test" })).not.toThrow();
    // A broken sink must not starve the healthy ones.
    expect(good).toHaveLength(1);
  });

  it("normalises non-Error throws", () => {
    const event = reportError("plain string failure", { surface: "test" });
    expect(event.message).toBe("plain string failure");
    expect(event.name).toBe("StringError");
  });

  it("unregisters a sink when its teardown is called", () => {
    const events: TelemetryEvent[] = [];
    const remove = registerTelemetrySink((e) => events.push(e));
    remove();
    reportError(new Error("after removal"), { surface: "test" });
    expect(events).toHaveLength(0);
  });
});

describe("installGlobalErrorHandlers", () => {
  beforeEach(() => {
    resetTelemetrySinks();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    resetTelemetrySinks();
    vi.restoreAllMocks();
  });

  it("reports unhandled promise rejections and stops after teardown", () => {
    const events: TelemetryEvent[] = [];
    registerTelemetrySink((e) => events.push(e));
    const teardown = installGlobalErrorHandlers(window);

    const event = new Event("unhandledrejection") as Event & { reason?: unknown };
    event.reason = new Error("dangling promise");
    window.dispatchEvent(event);

    expect(events).toHaveLength(1);
    expect(events[0].surface).toBe("window.unhandledrejection");

    teardown();
    window.dispatchEvent(event);
    expect(events).toHaveLength(1);
  });
});
