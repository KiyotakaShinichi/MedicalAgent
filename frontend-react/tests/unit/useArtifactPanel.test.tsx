import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { useArtifactPanel } from "../../src/pages/admin/sections/mle/useArtifactPanel";
import { resetTelemetrySinks } from "../../src/lib/telemetry";

describe("useArtifactPanel", () => {
  beforeEach(() => {
    resetTelemetrySinks();
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    resetTelemetrySinks();
    vi.restoreAllMocks();
  });

  it("loads the artifact on mount", async () => {
    const fetcher = vi.fn(async () => ({ status: "ok" }));
    const { result } = renderHook(() => useArtifactPanel(fetcher, undefined, "test"));

    await waitFor(() => expect(result.current.report).toEqual({ status: "ok" }));
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("surfaces a load failure as the panel error", async () => {
    const fetcher = vi.fn(async () => {
      throw new Error("artifact endpoint down");
    });
    const { result } = renderHook(() => useArtifactPanel(fetcher, undefined, "test"));

    await waitFor(() => expect(result.current.error).toBe("artifact endpoint down"));
    expect(result.current.report).toBeNull();
  });

  it("regenerates then reloads when refreshed", async () => {
    const order: string[] = [];
    const runner = vi.fn(async () => void order.push("run"));
    const fetcher = vi.fn(async () => {
      order.push("fetch");
      return { status: "ok" };
    });

    const { result } = renderHook(() => useArtifactPanel(fetcher, runner, "test"));
    await waitFor(() => expect(result.current.report).not.toBeNull());
    order.length = 0;

    await act(async () => {
      result.current.onRefresh();
    });

    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
    expect(order).toEqual(["run", "fetch"]);
  });

  it("reports a failed regeneration without discarding the loaded artifact", async () => {
    const fetcher = vi.fn(async () => ({ status: "ok" }));
    const runner = vi.fn(async () => {
      throw new Error("regeneration job failed");
    });

    const { result } = renderHook(() => useArtifactPanel(fetcher, runner, "test"));
    await waitFor(() => expect(result.current.report).toEqual({ status: "ok" }));

    await act(async () => {
      result.current.onRefresh();
    });

    await waitFor(() => expect(result.current.error).toBe("regeneration job failed"));
    // The stale-but-real artifact stays on screen; the notice says it is stale.
    expect(result.current.report).toEqual({ status: "ok" });
    expect(result.current.running).toBe(false);
  });

  it("lets a run failure supersede an earlier load failure", async () => {
    // The run is the more recent signal, so it is the one worth showing.
    const fetcher = vi.fn(async () => {
      throw new Error("load failed");
    });
    const runner = vi.fn(async () => {
      throw new Error("run failed");
    });

    const { result } = renderHook(() => useArtifactPanel(fetcher, runner, "test"));
    await waitFor(() => expect(result.current.error).toBe("load failed"));

    await act(async () => {
      result.current.onRefresh();
    });

    await waitFor(() => expect(result.current.error).toBe("run failed"));
  });

  it("degrades refresh to a plain refetch for read-only panels", async () => {
    const fetcher = vi.fn(async () => ({ status: "ok" }));
    const { result } = renderHook(() => useArtifactPanel(fetcher, undefined, "test"));
    await waitFor(() => expect(result.current.report).not.toBeNull());

    await act(async () => {
      result.current.refetch();
    });

    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
    expect(result.current.error).toBeNull();
  });

  it("exposes running while a regeneration is in flight", async () => {
    let release: (() => void) | undefined;
    const fetcher = vi.fn(async () => ({ status: "ok" }));
    const runner = vi.fn(() => new Promise<void>((resolve) => { release = resolve; }));

    const { result } = renderHook(() => useArtifactPanel(fetcher, runner, "test"));
    await waitFor(() => expect(result.current.report).not.toBeNull());

    act(() => {
      result.current.onRefresh();
    });
    await waitFor(() => expect(result.current.running).toBe(true));

    await act(async () => {
      release?.();
    });
    await waitFor(() => expect(result.current.running).toBe(false));
  });
});
