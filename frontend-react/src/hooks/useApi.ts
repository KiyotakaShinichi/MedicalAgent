import { useState, useEffect, useCallback, useRef } from "react";

type Status = "idle" | "loading" | "success" | "error";

interface UseApiResult<T> {
  data: T | null;
  status: Status;
  error: string | null;
  refetch: () => Promise<void>;
  /** Epoch ms of the last successful response. null until first success. */
  lastFetchedAt: number | null;
}

export function useApi<T>(
  fn: () => Promise<T>,
  _deps: unknown[] = []
): UseApiResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [lastFetchedAt, setLastFetchedAt] = useState<number | null>(null);
  const fnRef = useRef(fn);
  const depsRef = useRef<unknown[] | null>(null);
  const requestIdRef = useRef(0);
  const mountedRef = useRef(false);

  const run = useCallback(async () => {
    if (!mountedRef.current) return;
    const requestId = ++requestIdRef.current;
    setStatus("loading");
    setError(null);
    try {
      const nextData = await fnRef.current();
      if (!mountedRef.current || requestId !== requestIdRef.current) return;
      setData(nextData);
      setStatus("success");
      setLastFetchedAt(Date.now());
    } catch (error: unknown) {
      if (!mountedRef.current || requestId !== requestIdRef.current) return;
      setError(error instanceof Error ? error.message : String(error));
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    fnRef.current = fn;
  }, [fn]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      requestIdRef.current += 1;
      depsRef.current = null;
    };
  }, []);

  useEffect(() => {
    const previousDeps = depsRef.current;
    const dependenciesChanged = previousDeps === null
      || previousDeps.length !== _deps.length
      || _deps.some((value, index) => !Object.is(value, previousDeps[index]));

    if (!dependenciesChanged) return;
    depsRef.current = [..._deps];
    void run();
  });

  return { data, status, error, refetch: run, lastFetchedAt };
}
