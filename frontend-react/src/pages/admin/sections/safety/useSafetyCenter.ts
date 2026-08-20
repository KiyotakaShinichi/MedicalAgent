import { useCallback, useEffect, useRef, useState } from "react";
import {
  getSafetyCenter,
  getLlmJudgeEval,
  getMultilingualRefusalEval,
  runLlmJudgeEval,
  runDriftReport,
  runMultilingualRefusalEval,
  runRagEvalArtifact,
  runSafetyRedTeam,
} from "../../../../api/client";
import { reportError, reportExpectedError } from "../../../../lib/telemetry";
import type {
  LlmJudgeEval,
  MultilingualRefusalEval,
  SafetyCenter,
} from "../../../../types/api";

export type SafetyCenterStatus = "idle" | "loading" | "success" | "error";

/** Artifact regeneration jobs the admin can trigger from this surface. */
export type RegenerateKind = "safety" | "rag" | "drift";
export type ExtraEvalKind = "multilingual" | "llm_judge";

export interface UseSafetyCenterResult {
  data: SafetyCenter | null;
  multilingual: MultilingualRefusalEval | null;
  llmJudge: LlmJudgeEval | null;
  status: SafetyCenterStatus;
  /** Fatal load error — the section cannot render without the payload. */
  error: string | null;
  /**
   * Non-fatal error from a regeneration action. Kept separate from `error`
   * because the already-loaded artifacts stay valid and visible when a re-run
   * fails; conflating the two previously made action failures invisible.
   */
  actionError: string | null;
  dismissActionError: () => void;
  /** Key of the currently running job, or null. */
  running: string | null;
  reload: () => Promise<void>;
  regenerate: (kind: RegenerateKind, liveAgent?: boolean) => Promise<void>;
  runExtraEval: (kind: ExtraEvalKind) => Promise<void>;
}

const SURFACE = "admin.SafetyCenter";

/**
 * Owns every network interaction for the Safety & Evaluation Center.
 *
 * Splitting this out of the component means the presentational blocks are pure
 * functions of their props and can be tested without mocking the API client.
 *
 * Concurrency: `loadIdRef` fences stale responses. Without it, a slow initial
 * load resolving after a fast re-run would overwrite fresh artifacts with
 * stale ones — a correctness problem on a surface whose entire purpose is
 * reporting current safety posture.
 */
export function useSafetyCenter(): UseSafetyCenterResult {
  const [data, setData] = useState<SafetyCenter | null>(null);
  const [multilingual, setMultilingual] = useState<MultilingualRefusalEval | null>(null);
  const [llmJudge, setLlmJudge] = useState<LlmJudgeEval | null>(null);
  // Starts at "loading", not "idle": a fetch is scheduled unconditionally on
  // mount, and reporting "idle" for that first frame made the section render
  // its "No safety center data" empty state before the request had even begun.
  const [status, setStatus] = useState<SafetyCenterStatus>("loading");
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [running, setRunning] = useState<string | null>(null);

  const mountedRef = useRef(true);
  const loadIdRef = useRef(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      // Invalidate any in-flight load so its resolution is discarded.
      loadIdRef.current += 1;
    };
  }, []);

  const load = useCallback(async () => {
    const loadId = ++loadIdRef.current;
    setStatus("loading");
    setError(null);
    try {
      const result = await getSafetyCenter();
      if (!mountedRef.current || loadId !== loadIdRef.current) return;
      setData(result);

      // The two optional evals are allowed to be unavailable — a disabled LLM
      // judge or a missing multilingual artifact must not fail the whole page.
      const [multiResult, judgeResult] = await Promise.allSettled([
        getMultilingualRefusalEval(),
        getLlmJudgeEval(),
      ]);
      if (!mountedRef.current || loadId !== loadIdRef.current) return;

      if (multiResult.status === "fulfilled") {
        setMultilingual(multiResult.value);
      } else {
        reportExpectedError(multiResult.reason, `${SURFACE}.multilingual`);
      }
      if (judgeResult.status === "fulfilled") {
        setLlmJudge(judgeResult.value);
      } else {
        reportExpectedError(judgeResult.reason, `${SURFACE}.llmJudge`);
      }
      setStatus("success");
    } catch (e) {
      if (!mountedRef.current || loadId !== loadIdRef.current) return;
      reportError(e, { surface: `${SURFACE}.load` });
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }, []);

  // Kick the initial fetch off a macrotask rather than synchronously in the
  // effect body. `load` sets state on its first line, and doing that inline
  // makes React re-render before the effect phase finishes (the pattern
  // react-hooks/set-state-in-effect flags). The timeout also gives the cleanup
  // a chance to cancel the fetch entirely on an immediate unmount.
  useEffect(() => {
    const handle = window.setTimeout(() => {
      void load();
    }, 0);
    return () => window.clearTimeout(handle);
  }, [load]);

  const regenerate = useCallback(
    async (kind: RegenerateKind, liveAgent = false) => {
      const runKey = liveAgent ? `${kind}-live` : kind;
      setRunning(runKey);
      setActionError(null);
      try {
        if (kind === "safety") await runSafetyRedTeam(liveAgent);
        else if (kind === "rag") await runRagEvalArtifact(liveAgent);
        else await runDriftReport();
        await load();
      } catch (e) {
        if (!mountedRef.current) return;
        reportError(e, { surface: `${SURFACE}.regenerate`, detail: { kind, liveAgent } });
        setActionError(e instanceof Error ? e.message : String(e));
      } finally {
        if (mountedRef.current) setRunning(null);
      }
    },
    [load],
  );

  const runExtraEval = useCallback(async (kind: ExtraEvalKind) => {
    setRunning(kind);
    setActionError(null);
    try {
      if (kind === "multilingual") {
        const response = await runMultilingualRefusalEval();
        if (mountedRef.current) setMultilingual(response.result);
      } else {
        const response = await runLlmJudgeEval(30);
        if (mountedRef.current) setLlmJudge(response.result);
      }
    } catch (e) {
      if (!mountedRef.current) return;
      reportError(e, { surface: `${SURFACE}.runExtraEval`, detail: { kind } });
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      if (mountedRef.current) setRunning(null);
    }
  }, []);

  const dismissActionError = useCallback(() => setActionError(null), []);

  return {
    data,
    multilingual,
    llmJudge,
    status,
    error,
    actionError,
    dismissActionError,
    running,
    reload: load,
    regenerate,
    runExtraEval,
  };
}
