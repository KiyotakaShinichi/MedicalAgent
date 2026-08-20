import { useCallback, useEffect, useRef, useState } from "react";
import { getMyReportEnrichment } from "../api/client";
import { reportExpectedError } from "../lib/telemetry";

export type EnrichmentStatus = "idle" | "loading" | "success" | "error";

/** Minimum gap between polls, regardless of what the backend suggests. */
const MIN_RETRY_MS = 500;
/** Used when the backend omits `retry_after_ms`. */
const DEFAULT_RETRY_MS = 750;
/** A cold start can fail once while model artifacts load; allow a few retries. */
const TRANSIENT_FAILURE_RETRIES = 3;
const TRANSIENT_RETRY_MS = 1500;
/** Hard ceiling so a permanently-pending job cannot poll forever. */
const MAX_ATTEMPTS = 60;

export interface UseReportEnrichmentResult<T> {
  enrichment: T | null;
  status: EnrichmentStatus;
  /** Epoch ms of the successful load, or null. */
  fetchedAt: number | null;
  /** Begins polling. Safe to call repeatedly — only the first call starts a run. */
  start: () => void;
  /** Cancels polling and clears state so the next `start()` runs fresh. */
  reset: () => void;
}

/**
 * Polls the patient report's slow enrichment job.
 *
 * The patient report is served in two parts: core records return immediately,
 * and the synthetic engineering detail (model signals, hybrid prediction) is
 * computed by a background job. This hook owns the polling state machine for
 * the second half.
 *
 * Why it is a hook and not inline in the dashboard: the machine has four
 * distinct behaviours — bounded transient-failure recovery, backend-suggested
 * backoff with a floor, an attempt ceiling, and timer cleanup — and none of it
 * was reachable by a test while it lived in a 590-line component body.
 *
 * **Safety contract:** a failed or still-pending enrichment must leave
 * `enrichment` as `null`. The dashboard renders core records either way; it
 * must never present partially-computed model output as a finished result.
 */
export function useReportEnrichment<
  T extends {
    // Nullable as well as optional: the API models an unset retry hint as
    // explicit null, and the `??` below has to treat both the same way.
    report_enrichment?: { status?: string | null; retry_after_ms?: number | null };
  },
>(): UseReportEnrichmentResult<T> {
  const [enrichment, setEnrichment] = useState<T | null>(null);
  const [status, setStatus] = useState<EnrichmentStatus>("idle");
  const [fetchedAt, setFetchedAt] = useState<number | null>(null);

  const startedRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const mountedRef = useRef(true);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      timerRef.current = null;
    };
  }, []);

  const start = useCallback(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    // Deferred so callers can invoke this from an effect body without
    // triggering a synchronous cascading render.
    queueMicrotask(() => {
      if (mountedRef.current) setStatus("loading");
    });

    const schedule = (attempt: number, delayMs: number) => {
      timerRef.current = window.setTimeout(() => {
        void poll(attempt);
      }, delayMs);
    };

    const poll = async (attempt: number) => {
      if (!mountedRef.current) return;
      try {
        const value = (await getMyReportEnrichment()) as T;
        // The request can resolve after unmount; dropping it here is what
        // keeps this from writing to a torn-down component.
        if (!mountedRef.current) return;

        const jobStatus = value.report_enrichment?.status;

        if (jobStatus === "complete") {
          setEnrichment(value);
          setStatus("success");
          setFetchedAt(Date.now());
          return;
        }

        if (jobStatus === "failed") {
          if (attempt < TRANSIENT_FAILURE_RETRIES) {
            // The backend reschedules failed jobs on the next poll.
            schedule(attempt + 1, TRANSIENT_RETRY_MS);
            return;
          }
          reportExpectedError(
            new Error("Report enrichment job reported failed"),
            "patient.reportEnrichment",
            { attempt },
          );
          setStatus("error");
          return;
        }

        if (attempt >= MAX_ATTEMPTS) {
          reportExpectedError(
            new Error("Report enrichment did not complete within the attempt ceiling"),
            "patient.reportEnrichment",
            { attempt },
          );
          setStatus("error");
          return;
        }

        schedule(attempt + 1, Math.max(MIN_RETRY_MS, value.report_enrichment?.retry_after_ms ?? DEFAULT_RETRY_MS));
      } catch (error) {
        if (!mountedRef.current) return;
        if (attempt < TRANSIENT_FAILURE_RETRIES) {
          schedule(attempt + 1, TRANSIENT_RETRY_MS);
          return;
        }
        reportExpectedError(error, "patient.reportEnrichment", { attempt });
        setStatus("error");
      }
    };

    void poll(0);
  }, []);

  const reset = useCallback(() => {
    clearTimer();
    startedRef.current = false;
    setEnrichment(null);
    setStatus("idle");
    setFetchedAt(null);
  }, [clearTimer]);

  return { enrichment, status, fetchedAt, start, reset };
}
