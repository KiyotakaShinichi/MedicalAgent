import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

// The hook imports the API client eagerly, so the mock must be declared before
// the module under test is imported.
vi.mock("../../src/api/client", () => ({
  API_BASE: "http://test.local",
  getSafetyCenter: vi.fn(),
  getMultilingualRefusalEval: vi.fn(),
  getLlmJudgeEval: vi.fn(),
  runSafetyRedTeam: vi.fn(),
  runRagEvalArtifact: vi.fn(),
  runDriftReport: vi.fn(),
  runMultilingualRefusalEval: vi.fn(),
  runLlmJudgeEval: vi.fn(),
}));

import * as api from "../../src/api/client";
import { useSafetyCenter } from "../../src/pages/admin/sections/safety/useSafetyCenter";
import { resetTelemetrySinks } from "../../src/lib/telemetry";

const mocked = vi.mocked(api);

const CENTER = { safety_note: "synthetic only", generated_at: "2026-01-01T00:00:00Z" };

function primeSuccessfulLoad() {
  mocked.getSafetyCenter.mockResolvedValue(CENTER as never);
  mocked.getMultilingualRefusalEval.mockResolvedValue({ status: "passed" } as never);
  mocked.getLlmJudgeEval.mockResolvedValue({ status: "unavailable" } as never);
}

describe("useSafetyCenter", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetTelemetrySinks();
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    resetTelemetrySinks();
    vi.restoreAllMocks();
  });

  it("loads the center plus both optional evals", async () => {
    primeSuccessfulLoad();
    const { result } = renderHook(() => useSafetyCenter());

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.data).toEqual(CENTER);
    expect(result.current.multilingual).toEqual({ status: "passed" });
    expect(result.current.llmJudge).toEqual({ status: "unavailable" });
    expect(result.current.error).toBeNull();
  });

  it("surfaces a fatal error when the primary payload fails", async () => {
    mocked.getSafetyCenter.mockRejectedValue(new Error("backend down"));
    const { result } = renderHook(() => useSafetyCenter());

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toBe("backend down");
  });

  it("still renders when the optional evals are unavailable", async () => {
    // A disabled LLM judge or a missing multilingual artifact is an expected
    // state and must not take the whole safety dashboard down.
    mocked.getSafetyCenter.mockResolvedValue(CENTER as never);
    mocked.getMultilingualRefusalEval.mockRejectedValue(new Error("not generated"));
    mocked.getLlmJudgeEval.mockRejectedValue(new Error("adjudication disabled"));

    const { result } = renderHook(() => useSafetyCenter());

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.data).toEqual(CENTER);
    expect(result.current.multilingual).toBeNull();
    expect(result.current.llmJudge).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("reports a failed regeneration instead of silently dropping it", async () => {
    // Regression: the previous implementation wrote a failed re-run into the
    // same state as the fatal load error, which was only rendered in the
    // status === "error" branch. The message was set but never displayed, so
    // the button simply stopped spinning with no explanation.
    primeSuccessfulLoad();
    mocked.runSafetyRedTeam.mockRejectedValue(new Error("red-team runner exploded"));

    const { result } = renderHook(() => useSafetyCenter());
    await waitFor(() => expect(result.current.status).toBe("success"));

    await act(async () => {
      await result.current.regenerate("safety", false);
    });

    expect(result.current.actionError).toBe("red-team runner exploded");
    // The loaded artifacts stay valid and visible.
    expect(result.current.status).toBe("success");
    expect(result.current.data).toEqual(CENTER);
    expect(result.current.running).toBeNull();
  });

  it("clears the action error on dismissal and before the next run", async () => {
    primeSuccessfulLoad();
    mocked.runDriftReport.mockRejectedValue(new Error("drift failed"));

    const { result } = renderHook(() => useSafetyCenter());
    await waitFor(() => expect(result.current.status).toBe("success"));

    await act(async () => {
      await result.current.regenerate("drift");
    });
    expect(result.current.actionError).toBe("drift failed");

    act(() => result.current.dismissActionError());
    expect(result.current.actionError).toBeNull();

    mocked.runDriftReport.mockResolvedValue({ ok: true } as never);
    await act(async () => {
      await result.current.regenerate("drift");
    });
    expect(result.current.actionError).toBeNull();
  });

  it("reloads the payload after a successful regeneration", async () => {
    primeSuccessfulLoad();
    mocked.runRagEvalArtifact.mockResolvedValue({ ok: true } as never);

    const { result } = renderHook(() => useSafetyCenter());
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mocked.getSafetyCenter).toHaveBeenCalledTimes(1);

    await act(async () => {
      await result.current.regenerate("rag", true);
    });

    expect(mocked.runRagEvalArtifact).toHaveBeenCalledWith(true);
    expect(mocked.getSafetyCenter).toHaveBeenCalledTimes(2);
  });

  it("stores the result of an on-demand extra eval without a full reload", async () => {
    primeSuccessfulLoad();
    mocked.runMultilingualRefusalEval.mockResolvedValue({
      result: { status: "passed", summary: { pass_rate: 1 } },
    } as never);

    const { result } = renderHook(() => useSafetyCenter());
    await waitFor(() => expect(result.current.status).toBe("success"));

    await act(async () => {
      await result.current.runExtraEval("multilingual");
    });

    expect(result.current.multilingual).toEqual({ status: "passed", summary: { pass_rate: 1 } });
    expect(mocked.getSafetyCenter).toHaveBeenCalledTimes(1);
  });

  it("reports a failed extra eval as a non-fatal action error", async () => {
    primeSuccessfulLoad();
    mocked.runLlmJudgeEval.mockRejectedValue(new Error("no provider configured"));

    const { result } = renderHook(() => useSafetyCenter());
    await waitFor(() => expect(result.current.status).toBe("success"));

    await act(async () => {
      await result.current.runExtraEval("llm_judge");
    });

    expect(result.current.actionError).toBe("no provider configured");
    expect(result.current.status).toBe("success");
  });

  it("does not apply a stale load that resolves after unmount", async () => {
    let resolveLoad: ((value: unknown) => void) | undefined;
    mocked.getSafetyCenter.mockReturnValue(
      new Promise((resolve) => {
        resolveLoad = resolve;
      }) as never,
    );
    mocked.getMultilingualRefusalEval.mockResolvedValue({} as never);
    mocked.getLlmJudgeEval.mockResolvedValue({} as never);

    const { result, unmount } = renderHook(() => useSafetyCenter());
    unmount();

    await act(async () => {
      resolveLoad?.(CENTER);
    });

    // The fence in the hook discards the response rather than setting state on
    // an unmounted component.
    expect(result.current.data).toBeNull();
  });
});
