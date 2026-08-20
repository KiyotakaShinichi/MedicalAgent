import { AlertTriangle } from "lucide-react";

/**
 * Inline failure notice for one artifact panel.
 *
 * Replaces the single aggregated banner that previously sat at the top of
 * `MleSection` and showed only the first of fourteen possible runner errors.
 * Placing the notice next to the panel that failed tells the operator *which*
 * artifact is stale, which the aggregated version could not.
 *
 * The wording is deliberate: the panel below keeps rendering the previous
 * artifact, and the operator must not read stale evidence as current.
 */
export function PanelErrorNotice({ panel, error }: { panel: string; error: string | null }) {
  if (!error) return null;
  return (
    <div
      role="alert"
      className="flex items-start gap-2 px-3 py-2 rounded-lg border text-xs"
      style={{ background: "rgba(244,63,94,0.06)", borderColor: "rgba(244,63,94,0.28)", color: "var(--text)" }}
    >
      <AlertTriangle size={13} aria-hidden="true" style={{ flexShrink: 0, marginTop: 1 }} />
      <span>
        <strong>{panel} could not be updated.</strong> {error} Any values shown below are
        from the previous run and may be out of date.
      </span>
    </div>
  );
}
