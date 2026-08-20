import { useMemo } from "react";
import { useApi } from "../../../../hooks/useApi";
import { useArtifactRunner } from "../../../../hooks/useArtifactRunner";

export interface ArtifactPanelState<T> {
  /** Latest artifact, or null while loading / on failure. */
  report: T | null;
  /** True during the initial (or a re-triggered) GET. */
  loading: boolean;
  /** True while a regeneration job is in flight. */
  running: boolean;
  /** Load failure, regeneration failure, or null. */
  error: string | null;
  /** Regenerate then reload. Never rejects. */
  onRefresh: () => void;
  /** Reload without regenerating. */
  refetch: () => void;
}

const NO_OP_JOB = async () => {};

/**
 * Binds one admin evidence artifact to its own data and action state.
 *
 * `MleSection` previously held 24 `useApi` calls and 14 runner hooks in a
 * single component body, so no panel owned its own behaviour and a reader had
 * to trace a variable name across 600 lines to find which artifact it drove.
 * One call to this hook is the complete data contract for one panel, and the
 * returned shape matches the `{ report, loading, running, onRefresh }` props
 * the existing card components already accept.
 *
 * Load and regeneration failures are merged into a single `error`: from the
 * operator's point of view both mean "this panel is not showing you a current
 * artifact", and the panel has one place to say so.
 *
 * @param fetcher  GET for the artifact
 * @param runner   regeneration job; omit for read-only panels, where
 *                 `onRefresh` degrades to a plain refetch
 * @param surface  telemetry label, e.g. "admin.mle.leakageAudit"
 */
export function useArtifactPanel<T>(
  fetcher: () => Promise<T>,
  runner: (() => Promise<unknown>) | undefined,
  surface: string,
): ArtifactPanelState<T> {
  const { data, status, error: loadError, refetch } = useApi<T>(fetcher, []);

  // useArtifactRunner must be called unconditionally. A read-only panel gets a
  // no-op job, so pressing refresh simply reloads.
  const job = useMemo(() => runner ?? NO_OP_JOB, [runner]);
  const { running, error: runError, run } = useArtifactRunner(job, refetch, surface);

  return {
    report: data,
    loading: status === "loading",
    running,
    // A failed run is the more recent signal, so it wins over a stale load error.
    error: runError ?? (status === "error" ? loadError : null),
    onRefresh: run,
    refetch,
  };
}
