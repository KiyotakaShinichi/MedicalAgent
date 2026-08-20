import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { useArtifactRunner } from "../../src/hooks/useArtifactRunner";
import {
  registerTelemetrySink,
  resetTelemetrySinks,
  type TelemetryEvent,
} from "../../src/lib/telemetry";

describe("useArtifactRunner", () => {
  beforeEach(() => {
    resetTelemetrySinks();
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    resetTelemetrySinks();
    vi.restoreAllMocks();
  });

  it("runs the job then the reload callback", async () => {
    const order: string[] = [];
    const job = vi.fn(async () => void order.push("job"));
    const onDone = vi.fn(async () => void order.push("done"));

    const { result } = renderHook(() => useArtifactRunner(job, onDone, "test"));
    await act(async () => {
      await result.current.run();
    });

    expect(order).toEqual(["job", "done"]);
    expect(result.current.running).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("reports running while the job is in flight", async () => {
    let release: (() => void) | undefined;
    const job = vi.fn(() => new Promise<void>((resolve) => { release = resolve; }));
    const { result } = renderHook(() => useArtifactRunner(job, () => {}, "test"));

    act(() => {
      void result.current.run();
    });
    await waitFor(() => expect(result.current.running).toBe(true));

    await act(async () => {
      release?.();
    });
    await waitFor(() => expect(result.current.running).toBe(false));
  });

  it("captures a failed run instead of leaving an unhandled rejection", async () => {
    // Regression: the hand-written version used try/finally with no catch, so
    // a rejected job reset the spinner and then escaped as an unhandled
    // rejection with nothing shown to the operator.
    const job = vi.fn(async () => {
      throw new Error("artifact job crashed");
    });
    const onDone = vi.fn();

    const { result } = renderHook(() => useArtifactRunner(job, onDone, "test"));
    await act(async () => {
      await expect(result.current.run()).resolves.toBeUndefined();
    });

    expect(result.current.error).toBe("artifact job crashed");
    expect(result.current.running).toBe(false);
  });

  it("does not reload after a failed run", async () => {
    // Refetching would re-read the unchanged artifact and imply success.
    const job = vi.fn(async () => {
      throw new Error("failed");
    });
    const onDone = vi.fn();

    const { result } = renderHook(() => useArtifactRunner(job, onDone, "test"));
    await act(async () => {
      await result.current.run();
    });

    expect(onDone).not.toHaveBeenCalled();
  });

  it("reports the failure to telemetry once", async () => {
    const events: TelemetryEvent[] = [];
    registerTelemetrySink((e) => events.push(e));

    const { result } = renderHook(() =>
      useArtifactRunner(async () => { throw new Error("boom"); }, () => {}, "admin.mle.leakageAudit"),
    );
    await act(async () => {
      await result.current.run();
    });

    expect(events).toHaveLength(1);
    expect(events[0].surface).toBe("admin.mle.leakageAudit");
    expect(events[0].kind).toBe("unexpected");
  });

  it("clears the previous error when a new run starts and on explicit dismissal", async () => {
    const job = vi.fn()
      .mockRejectedValueOnce(new Error("first failed"))
      .mockResolvedValueOnce(undefined);

    const { result } = renderHook(() => useArtifactRunner(job, () => {}, "test"));

    await act(async () => { await result.current.run(); });
    expect(result.current.error).toBe("first failed");

    act(() => result.current.clearError());
    expect(result.current.error).toBeNull();

    await act(async () => { await result.current.run(); });
    expect(result.current.error).toBeNull();
  });

  it("keeps a stable run identity across re-renders", () => {
    const { result, rerender } = renderHook(() => useArtifactRunner(async () => {}, () => {}, "test"));
    const first = result.current.run;
    rerender();
    expect(result.current.run).toBe(first);
  });

  it("does not set state after unmount", async () => {
    let release: (() => void) | undefined;
    const job = vi.fn(() => new Promise<void>((resolve) => { release = resolve; }));

    const { result, unmount } = renderHook(() => useArtifactRunner(job, () => {}, "test"));
    act(() => {
      void result.current.run();
    });
    unmount();

    await act(async () => {
      release?.();
    });
    // No "state update on unmounted component" warning should be produced.
    expect(console.error).not.toHaveBeenCalled();
  });
});
