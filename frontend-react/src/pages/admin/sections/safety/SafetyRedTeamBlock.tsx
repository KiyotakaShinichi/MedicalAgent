import { Badge } from "../../../../components/ui/Badge";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { EmptyPane, ErrorPane } from "../../../../components/ui/Spinner";
import { EvalIntegrityFooter } from "../../../../components/ui/EvalIntegrityFooter";
import type { SafetyRedTeamArtifact } from "../../../../types/api";
import { coerceIntegrityStatus, fmtRate, statusBadge } from "./safetyFormat";

/** How many failed cases to list before deferring to the raw artifact. */
const MAX_LISTED_FAILURES = 8;

/**
 * Red-team refusal benchmark results.
 *
 * Renders three mutually exclusive states — not generated, artifact error, and
 * summary — so a missing artifact never renders as a zeroed-out pass board.
 */
export function SafetyRedTeamBlock({ artifact }: { artifact: SafetyRedTeamArtifact }) {
  if (!artifact || artifact.status === "not_generated") {
    return <EmptyPane label="Safety red-team artifact not generated yet. Click 'Re-run' to produce it." />;
  }
  if (artifact.status === "error") {
    return <ErrorPane message={artifact.message ?? "Artifact error"} />;
  }
  const summary = artifact.summary;
  if (!summary) return <EmptyPane label="Artifact missing summary block." />;

  const cases = artifact.cases ?? [];
  const failed = cases.filter((c) => !c.pass);
  const totalCases = summary.total_cases ?? cases.length;
  const failedCount = summary.failed_cases?.length ?? failed.length;

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          label="Pass rate"
          value={fmtRate(summary.pass_rate)}
          status={statusBadge(summary.status)}
          sub={`${totalCases - failedCount}/${totalCases} passed`}
        />
        <MetricCard
          label="Failed cases"
          value={failedCount}
          status={failedCount ? "red" : "green"}
        />
        <MetricCard
          label="Categories"
          value={Object.keys(summary.category_counts ?? {}).length}
          status="muted"
        />
        <MetricCard
          label="Refusal types"
          value={Object.keys(summary.refusal_type_counts ?? {}).length}
          status="muted"
        />
      </div>

      {failed.length > 0 && (
        <section aria-label="Failed red-team cases">
          <p className="text-xs font-semibold mb-2" style={{ color: "var(--rose)" }}>
            {failed.length} failed case{failed.length !== 1 ? "s" : ""}
          </p>
          {failed.slice(0, MAX_LISTED_FAILURES).map((c) => (
            <div
              key={c.case_id}
              className="py-2 border-b last:border-0"
              style={{ borderColor: "var(--border)" }}
            >
              <div className="flex items-center gap-2 mb-1">
                <Badge variant="red">{c.category}</Badge>
                <span className="text-xs font-medium" style={{ color: "var(--text)" }}>
                  {c.case_id}
                </span>
              </div>
              <p className="text-xs" style={{ color: "var(--text-dim)" }}>
                {c.input_message}
              </p>
              {c.reason && (
                <p className="text-xs" style={{ color: "var(--rose)" }}>
                  {c.reason}
                </p>
              )}
            </div>
          ))}
          {failed.length > MAX_LISTED_FAILURES && (
            <p className="text-xs mt-2" style={{ color: "var(--text-faint)" }}>
              Showing {MAX_LISTED_FAILURES} of {failed.length}. See the artifact for the full list.
            </p>
          )}
        </section>
      )}

      <EvalIntegrityFooter
        totalN={totalCases}
        passCount={totalCases - failedCount}
        failCount={failedCount}
        skippedCount={0}
        authorship="internal"
        clinicalValidation={false}
        wasUsedForTuning={false}
        status={coerceIntegrityStatus(summary.status)}
        artifactPath="Data/evals/safety/latest_safety_red_team.json"
        caveat="Internal red-team bank — refusal benchmark only. Coverage of the long tail of unsafe prompts is not implied."
      />
    </div>
  );
}
