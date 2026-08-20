import { MetricCard } from "../../../../components/ui/MetricCard";
import { EmptyPane, ErrorPane } from "../../../../components/ui/Spinner";
import type { RagEvalArtifact } from "../../../../types/api";
import { fmtRate, fmtScore, statusBadge, type BadgeTone } from "./safetyFormat";

/**
 * Two-state tone used across this block: a metric either clears its threshold
 * or it is flagged for attention. Absent metrics stay amber rather than green —
 * an unmeasured RAG guarantee is not a met guarantee.
 */
function passOrWatch(met: boolean): BadgeTone {
  return met ? "green" : "amber";
}

const atLeast = (value: number | null | undefined, threshold: number) =>
  value !== null && value !== undefined && value >= threshold;

const atMost = (value: number | null | undefined, threshold: number) =>
  value !== null && value !== undefined && value <= threshold;

/** Retrieval-augmented generation evaluation summary. */
export function RagEvalBlock({ artifact }: { artifact: RagEvalArtifact }) {
  if (!artifact || artifact.status === "not_generated") {
    return <EmptyPane label="RAG eval artifact not generated yet. Click 'Re-run' to produce it." />;
  }
  if (artifact.status === "error") {
    return <ErrorPane message={artifact.message ?? "Artifact error"} />;
  }
  const summary = artifact.summary;
  if (!summary) return <EmptyPane label="Artifact missing summary block." />;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <MetricCard
        label="Overall pass"
        value={fmtRate(summary.pass_rate)}
        status={statusBadge(summary.status)}
      />
      <MetricCard
        label="Citation coverage"
        value={fmtRate(summary.citation_coverage_rate)}
        status={passOrWatch(atLeast(summary.citation_coverage_rate, 0.9))}
      />
      <MetricCard
        label="Source hit"
        value={fmtRate(summary.expected_source_hit_rate)}
        status={passOrWatch(atLeast(summary.expected_source_hit_rate, 0.8))}
      />
      <MetricCard
        label="Refusal correct"
        value={fmtRate(summary.refusal_correct_rate)}
        status={passOrWatch(summary.refusal_correct_rate === 1)}
      />
      <MetricCard
        label="Grounding (avg)"
        value={fmtScore(summary.average_grounding_score, 2)}
        status="muted"
      />
      <MetricCard
        label="Hallucination (avg)"
        value={fmtScore(summary.average_hallucination_score, 2)}
        status={passOrWatch(atMost(summary.average_hallucination_score, 0.3))}
      />
      <MetricCard
        label="Retrieval P@3"
        value={fmtScore(summary.average_retrieval_precision_at_3, 2)}
        status="muted"
      />
      <MetricCard
        label="Rewrite quality"
        value={fmtRate(summary.rewrite_term_hit_rate)}
        status="muted"
      />
    </div>
  );
}
