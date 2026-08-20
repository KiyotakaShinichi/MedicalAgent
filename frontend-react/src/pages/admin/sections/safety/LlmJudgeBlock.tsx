import { MetricCard } from "../../../../components/ui/MetricCard";
import { EmptyPane } from "../../../../components/ui/Spinner";
import type { LlmJudgeEval } from "../../../../types/api";
import { fmtRate, fmtScore, statusBadge } from "./safetyFormat";

/**
 * Optional LLM-as-judge evaluation.
 *
 * The `unavailable` branch matters for the product's safety posture: when
 * adjudication is disabled or no provider is configured, the surface must say
 * so plainly. Rendering an empty metric board instead would let a disabled
 * evaluator read as a clean bill of health.
 */
export function LlmJudgeBlock({ artifact }: { artifact: LlmJudgeEval | null }) {
  if (!artifact || artifact.status === "not_generated") {
    return <EmptyPane label="No LLM-judge report has been generated yet." />;
  }
  if (artifact.status === "unavailable") {
    return (
      <div
        role="status"
        className="rounded-md border p-3 text-xs"
        style={{ background: "var(--surface2)", borderColor: "var(--border)", color: "var(--text-dim)" }}
      >
        {artifact.message ?? "LLM adjudication is disabled or no provider is configured."}
        <br />
        <span style={{ color: "var(--text-faint)" }}>{artifact.claim_boundary}</span>
      </div>
    );
  }
  const summary = artifact.summary;
  if (!summary) return <EmptyPane label="Artifact missing summary block." />;

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Status" value={artifact.status ?? "unknown"} status={statusBadge(artifact.status)} />
        <MetricCard label="Pass rate" value={fmtRate(summary.pass_rate)} status={summary.pass_rate === 1 ? "green" : "amber"} />
        <MetricCard label="Judge coverage" value={fmtRate(summary.coverage_rate)} status={summary.coverage_rate === 1 ? "green" : "amber"} />
        <MetricCard label="Groundedness" value={fmtScore(summary.average_groundedness_score)} status="muted" />
        <MetricCard label="Unsafe advice" value={fmtRate(summary.unsafe_medical_advice_rate)} status={summary.unsafe_medical_advice_rate ? "red" : "green"} />
      </div>
      <p className="text-xs" style={{ color: "var(--text-dim)" }}>
        Provider: {artifact.provider ?? "none"} {artifact.model ? `(${artifact.model})` : ""}. This is an optional LLM-as-judge heuristic.
      </p>
      {artifact.claim_boundary && (
        <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{artifact.claim_boundary}</p>
      )}
    </div>
  );
}
