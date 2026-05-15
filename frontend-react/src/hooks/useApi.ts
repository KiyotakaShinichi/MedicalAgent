import { useState, useEffect, useCallback } from "react";

type Status = "idle" | "loading" | "success" | "error";

interface UseApiResult<T> {
  data: T | null;
  status: Status;
  error: string | null;
  refetch: () => void;
  /** Epoch ms of the last successful response. null until first success. */
  lastFetchedAt: number | null;
}

export function useApi<T>(
  fn: () => Promise<T>,
  _deps: unknown[] = []
): UseApiResult<T> {
  void _deps;
  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [lastFetchedAt, setLastFetchedAt] = useState<number | null>(null);

  const run = useCallback(() => {
    setStatus("loading");
    setError(null);
    fn()
      .then((d) => {
        setData(d);
        setStatus("success");
        setLastFetchedAt(Date.now());
      })
      .catch((e: Error) => {
        setError(e.message);
        setStatus("error");
      });
  }, [fn]);

  useEffect(() => {
    void Promise.resolve().then(run);
  }, [run]);

  return { data, status, error, refetch: run, lastFetchedAt };
}
