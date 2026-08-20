import type { ReactNode } from "react";
import { Badge } from "../../../../components/ui/Badge";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { EmptyPane, ErrorPane } from "../../../../components/ui/Spinner";
import type { BenchmarkLadderSummary } from "../../../../types/api";
import { fmtRate, fmtScore, highIsGood, lowIsGood, statusBadge } from "./safetyFormat";

function BenchmarkGroup({
  title,
  status,
  children,
}: {
  title: string;
  status?: string;
  children: ReactNode;
}) {
  return (
    <section
      className="p-3 rounded-md border"
      style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
      aria-label={title}
    >
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs font-semibold m-0" style={{ color: "var(--text)" }}>
          {title}
        </h4>
        <Badge variant={statusBadge(status)}>{status ?? "n/a"}</Badge>
      </div>
      {children}
    </section>
  );
}

/**
 * Cross-cutting benchmark ladder: the six families of metrics that gate a
 * release. Every group renders even when its artifact is absent, so a missing
 * benchmark shows as "—" rather than silently disappearing from the board.
 */
export function BenchmarkLadderBlock({ artifact }: { artifact: BenchmarkLadderSummary | null }) {
  if (!artifact || artifact.status === "not_generated") {
    return <EmptyPane label="Benchmark ladder not generated yet. Run scripts/generate_benchmark_report.py." />;
  }
  if (artifact.status === "error") {
    return <ErrorPane message={artifact.message ?? "Benchmark ladder artifact error"} />;
  }

  const benchmarks = artifact.benchmarks ?? {};
  const { safety, adversarial, rag, model, realism, clinician_summary: clinician } = benchmarks;

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <BenchmarkGroup title="Safety benchmark" status={safety?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard label="Unsafe pass rate" value={fmtRate(safety?.unsafe_pass_rate)} status={lowIsGood(safety?.unsafe_pass_rate)} />
            <MetricCard label="Urgent escalation recall" value={fmtRate(safety?.urgent_escalation_recall)} status={highIsGood(safety?.urgent_escalation_recall)} />
            <MetricCard label="Privacy leak rate" value={fmtRate(safety?.privacy_leak_rate)} status={lowIsGood(safety?.privacy_leak_rate)} />
            <MetricCard label="Injection resistance" value={fmtRate(safety?.prompt_injection_resistance)} status={highIsGood(safety?.prompt_injection_resistance)} />
          </div>
        </BenchmarkGroup>

        <BenchmarkGroup title="Adversarial benchmark" status={adversarial?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard label="Attack block rate" value={fmtRate(adversarial?.attack_block_rate)} status={highIsGood(adversarial?.attack_block_rate)} />
          </div>
        </BenchmarkGroup>

        <BenchmarkGroup title="RAG benchmark" status={rag?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard label="Pass rate" value={fmtRate(rag?.pass_rate)} status={highIsGood(rag?.pass_rate)} />
            <MetricCard label="Citation precision" value={fmtRate(rag?.citation_coverage)} status={highIsGood(rag?.citation_coverage)} />
            <MetricCard label="Source hit" value={fmtRate(rag?.expected_source_hit)} status={highIsGood(rag?.expected_source_hit)} />
            <MetricCard label="Refusal correctness" value={fmtRate(rag?.refusal_correct)} status={highIsGood(rag?.refusal_correct)} />
            <MetricCard label="Unsafe answer rate" value={fmtRate(rag?.unsafe_answer_rate)} status={lowIsGood(rag?.unsafe_answer_rate, 0.0, 0.05)} />
          </div>
        </BenchmarkGroup>

        <BenchmarkGroup title="Model benchmark" status={model?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard label="AUROC" value={fmtScore(model?.synthetic_champion_auroc, 3)} status="muted" />
            <MetricCard label="AUPRC" value={fmtScore(model?.synthetic_champion_auprc, 3)} status="muted" />
            <MetricCard label="Brier" value={fmtScore(model?.synthetic_champion_brier, 3)} status="muted" />
            <MetricCard label="ECE (post-temp)" value={fmtScore(model?.synthetic_champion_ece_after, 3)} status="muted" />
          </div>
        </BenchmarkGroup>

        <BenchmarkGroup title="Synthetic realism" status={realism?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard label="Alignment score" value={fmtScore(realism?.alignment_score, 3)} status={statusBadge(realism?.status)} />
            <MetricCard label="Checks status" value={realism?.realism_checks_status ?? "—"} status={statusBadge(realism?.status)} />
          </div>
        </BenchmarkGroup>

        <BenchmarkGroup title="Clinician summary" status={clinician?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard label="Completeness" value={fmtRate(clinician?.summary_completeness_rate)} status={highIsGood(clinician?.summary_completeness_rate, 0.85, 0.7)} />
            <MetricCard label="Unsafe advice" value={fmtRate(clinician?.unsafe_advice_rate)} status={lowIsGood(clinician?.unsafe_advice_rate, 0.0, 0.05)} />
          </div>
        </BenchmarkGroup>
      </div>

      {(artifact.report_path || artifact.csv_path) && (
        <div className="text-xs" style={{ color: "var(--text-dim)" }}>
          {artifact.report_path && <span>Report: {artifact.report_path}</span>}
          {artifact.report_path && artifact.csv_path && <span> · </span>}
          {artifact.csv_path && <span>CSV: {artifact.csv_path}</span>}
        </div>
      )}
      {artifact.claim_boundary && (
        <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>
          {artifact.claim_boundary}
        </p>
      )}
    </div>
  );
}
