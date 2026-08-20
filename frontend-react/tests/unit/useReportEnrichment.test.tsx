import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

vi.mock("../../src/api/client", () => ({
  API_BASE: "http://test.local",
  getMyReportEnrichment: vi.fn(),
}));

import * as api from "../../src/api/client";
import { useReportEnrichment } from "../../src/hooks/useReportEnrichment";
import { resetTelemetrySinks } from "../../src/lib/telemetry";

const mocked = vi.mocked(api);

type Enrichment = {
  value?: string;
  report_enrichment?: { status?: string | null; retry_after_ms?: number | null };
};

/** Advance fake timers and flush the promise microtask queue between ticks. */
async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe("useReportEnrichment", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetTelemetrySinks();
    vi.useFakeTimers();
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.useRealTimers();
    resetTelemetrySinks();
    vi.restoreAllMocks();
  });

  it("starts idle and does not fetch until started", () => {
    const { result } = renderHook(() => useReportEnrichment<Enrichment>());
    expect(result.current.status).toBe("idle");
    expect(result.current.enrichment).toBeNull();
    expect(mocked.getMyReportEnrichment).not.toHaveBeenCalled();
  });

  it("resolves on the first poll when the job is already complete", async () => {
    mocked.getMyReportEnrichment.mockResolvedValue({
      value: "done",
      report_enrichment: { status: "complete" },
    } as never);

    const { result } = renderHook(() => useReportEnrichment<Enrichment>());
    act(() => result.current.start());
    await advance(0);

    expect(result.current.status).toBe("success");
    expect(result.current.enrichment).toEqual({ value: "done", report_enrichment: { status: "complete" } });
    expect(result.current.fetchedAt).toBeTypeOf("number");
  });

  it("keeps polling while the job is pending, honouring the backend's retry hint", async () => {
    mocked.getMyReportEnrichment
      .mockResolvedValueOnce({ report_enrichment: { status: "pending", retry_after_ms: 900 } } as never)
      .mockResolvedValueOnce({ report_enrichment: { status: "complete" } } as never);

    const { result } = renderHook(() => useReportEnrichment<Enrichment>());
    act(() => result.current.start());
    await advance(0);

    expect(mocked.getMyReportEnrichment).toHaveBeenCalledTimes(1);
    expect(result.current.status).toBe("loading");

    // Nothing should fire before the suggested delay elapses.
    await advance(800);
    expect(mocked.getMyReportEnrichment).toHaveBeenCalledTimes(1);

    await advance(200);
    expect(result.current.status).toBe("success");
  });

  it("floors an unreasonably small retry hint so it cannot hot-loop", async () => {
    mocked.getMyReportEnrichment
      .mockResolvedValueOnce({ report_enrichment: { status: "pending", retry_after_ms: 1 } } as never)
      .mockResolvedValue({ report_enrichment: { status: "complete" } } as never);

    const { result } = renderHook(() => useReportEnrichment<Enrichment>());
    act(() => result.current.start());
    await advance(0);

    await advance(400);
    expect(mocked.getMyReportEnrichment).toHaveBeenCalledTimes(1);

    await advance(150);
    expect(mocked.getMyReportEnrichment).toHaveBeenCalledTimes(2);
  });

  it("treats a null retry hint as the default delay", async () => {
    mocked.getMyReportEnrichment
      .mockResolvedValueOnce({ report_enrichment: { status: "pending", retry_after_ms: null } } as never)
      .mockResolvedValue({ report_enrichment: { status: "complete" } } as never);

    const { result } = renderHook(() => useReportEnrichment<Enrichment>());
    act(() => result.current.start());
    await advance(0);

    await advance(700);
    expect(mocked.getMyReportEnrichment).toHaveBeenCalledTimes(1);
    await advance(100);
    expect(mocked.getMyReportEnrichment).toHaveBeenCalledTimes(2);
  });

  it("recovers from a bounded number of transient job failures", async () => {
    // A cold start can report "failed" once while model artifacts load.
    mocked.getMyReportEnrichment
      .mockResolvedValueOnce({ report_enrichment: { status: "failed" } } as never)
      .mockResolvedValueOnce({ report_enrichment: { status: "complete" } } as never);

    const { result } = renderHook(() => useReportEnrichment<Enrichment>());
    act(() => result.current.start());
    await advance(0);
    expect(result.current.status).toBe("loading");

    await advance(1500);
    expect(result.current.status).toBe("success");
  });

  it("gives up after repeated job failures and never exposes partial data", async () => {
    mocked.getMyReportEnrichment.mockResolvedValue({ report_enrichment: { status: "failed" } } as never);

    const { result } = renderHook(() => useReportEnrichment<Enrichment>());
    act(() => result.current.start());
    await advance(0);
    await advance(1500 * 4);

    expect(result.current.status).toBe("error");
    // Safety contract: a failed enrichment must leave the caller with null so
    // the dashboard renders core records only, never half-computed model output.
    expect(result.current.enrichment).toBeNull();
  });

  it("recovers from transient network rejections", async () => {
    mocked.getMyReportEnrichment
      .mockRejectedValueOnce(new Error("network blip"))
      .mockResolvedValueOnce({ report_enrichment: { status: "complete" } } as never);

    const { result } = renderHook(() => useReportEnrichment<Enrichment>());
    act(() => result.current.start());
    await advance(0);
    await advance(1500);

    expect(result.current.status).toBe("success");
  });

  it("errors out after repeated network rejections", async () => {
    mocked.getMyReportEnrichment.mockRejectedValue(new Error("backend down"));

    const { result } = renderHook(() => useReportEnrichment<Enrichment>());
    act(() => result.current.start());
    await advance(0);
    await advance(1500 * 4);

    expect(result.current.status).toBe("error");
    expect(result.current.enrichment).toBeNull();
  });

  it("ignores repeat start calls while a run is in flight", async () => {
    mocked.getMyReportEnrichment.mockResolvedValue({ report_enrichment: { status: "complete" } } as never);

    const { result } = renderHook(() => useReportEnrichment<Enrichment>());
    act(() => {
      result.current.start();
      result.current.start();
      result.current.start();
    });
    await advance(0);

    expect(mocked.getMyReportEnrichment).toHaveBeenCalledTimes(1);
  });

  it("reset clears state and permits a fresh run", async () => {
    mocked.getMyReportEnrichment.mockResolvedValue({ report_enrichment: { status: "complete" } } as never);

    const { result } = renderHook(() => useReportEnrichment<Enrichment>());
    act(() => result.current.start());
    await advance(0);
    expect(result.current.status).toBe("success");

    act(() => result.current.reset());
    expect(result.current.status).toBe("idle");
    expect(result.current.enrichment).toBeNull();
    expect(result.current.fetchedAt).toBeNull();

    act(() => result.current.start());
    await advance(0);
    expect(result.current.status).toBe("success");
    expect(mocked.getMyReportEnrichment).toHaveBeenCalledTimes(2);
  });

  it("stops polling after unmount", async () => {
    mocked.getMyReportEnrichment.mockResolvedValue({
      report_enrichment: { status: "pending", retry_after_ms: 500 },
    } as never);

    const { result, unmount } = renderHook(() => useReportEnrichment<Enrichment>());
    act(() => result.current.start());
    await advance(0);
    expect(mocked.getMyReportEnrichment).toHaveBeenCalledTimes(1);

    unmount();
    await advance(5000);

    // The pending timer was cleared, so no further polls were issued.
    expect(mocked.getMyReportEnrichment).toHaveBeenCalledTimes(1);
  });
});
