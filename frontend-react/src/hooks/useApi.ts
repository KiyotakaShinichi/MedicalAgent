import { useState, useEffect, useCallback } from "react";

type Status = "idle" | "loading" | "success" | "error";

export function useApi<T>(
  fn: () => Promise<T>,
  _deps: unknown[] = []
): { data: T | null; status: Status; error: string | null; refetch: () => void } {
  void _deps;
  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(() => {
    setStatus("loading");
    setError(null);
    fn()
      .then((d) => {
        setData(d);
        setStatus("success");
      })
      .catch((e: Error) => {
        setError(e.message);
        setStatus("error");
      });
  }, [fn]);

  useEffect(() => {
    void Promise.resolve().then(run);
  }, [run]);

  return { data, status, error, refetch: run };
}
