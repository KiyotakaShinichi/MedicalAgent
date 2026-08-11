import { act, renderHook, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { describe, expect, it, vi } from "vitest";

import { useApi } from "../../src/hooks/useApi";

describe("useApi", () => {
  it("does not refetch just because an inline request function is recreated", async () => {
    const request = vi.fn(async (value: number) => ({ value }));
    const { result, rerender } = renderHook(
      ({ value }) => useApi(() => request(value), [value]),
      { initialProps: { value: 1 } },
    );

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.data).toEqual({ value: 1 });
    expect(request).toHaveBeenCalledTimes(1);

    rerender({ value: 1 });
    await act(async () => undefined);
    expect(request).toHaveBeenCalledTimes(1);

    rerender({ value: 2 });
    await waitFor(() => expect(result.current.data).toEqual({ value: 2 }));
    expect(request).toHaveBeenCalledTimes(2);

    await act(async () => {
      await result.current.refetch();
    });
    expect(request).toHaveBeenCalledTimes(3);
  });

  it("settles after React StrictMode replays mount effects", async () => {
    const request = vi.fn(async () => ({ ready: true }));
    const { result } = renderHook(
      () => useApi(() => request(), []),
      { wrapper: StrictMode },
    );

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.data).toEqual({ ready: true });
    expect(request).toHaveBeenCalledTimes(2);
  });
});
