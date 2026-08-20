import { useCallback, useEffect, useRef, useState } from "react";
import { reportError } from "../lib/telemetry";

export interface UseArtifactRunnerResult {
  /** True while the job is in flight — bind to a Button's `loading` prop. */
  running: boolean;
  /** Message from the last failed run, or null. Cleared when a run starts. */
  error: string | null;
  clearError: () => void;
  /** Starts the job. Never rejects, so `onClick={() => void run()}` is safe. */
  run: () => Promise<void>;
}

/**
 * Runs a "regenerate this artifact, then reload it" admin action.
 *
 * The admin dashboard has a dozen of these, and each was previously written
 * out by hand as a `useState` flag plus a `try/finally`. That shape has a real
 * defect, not just duplication: with a `finally` and no `catch`, a rejected
 * run resets the spinner and then propagates as an unhandled promise
 * rejection. The operator sees the button stop spinning and nothing else — no
 * error, and no indication that the artifact on screen is still the old one.
 *
 * This hook keeps the spinner behaviour, captures the failure so the caller
 * can render it, and reports it once to telemetry.
 *
 * @param job      the mutation that regenerates the artifact
 * @param onDone   reload callback, typically a `useApi` refetch. Only runs on
 *                 success — refetching after a failure would just re-read the
 *                 unchanged artifact and imply the run had worked.
 * @param surface  telemetry label, e.g. "admin.mle.leakageAudit"
 */
export function useArtifactRunner(
  job: () => Promise<unknown>,
  onDone: () => void | Promise<unknown>,
  surface: string,
): UseArtifactRunnerResult {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Latest-value refs keep `run` stable across renders, so passing it straight
  // to an onClick does not invalidate memoised children every render.
  const jobRef = useRef(job);
  const doneRef = useRef(onDone);
  const mountedRef = useRef(true);

  useEffect(() => {
    jobRef.current = job;
    doneRef.current = onDone;
  });

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      await jobRef.current();
      await doneRef.current();
    } catch (e) {
      reportError(e, { surface, kind: "unexpected" });
      if (mountedRef.current) {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      if (mountedRef.current) setRunning(false);
    }
  }, [surface]);

  const clearError = useCallback(() => setError(null), []);

  return { running, error, clearError, run };
}
