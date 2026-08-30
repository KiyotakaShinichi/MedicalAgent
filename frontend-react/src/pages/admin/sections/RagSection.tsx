import { useApi } from "../../../hooks/useApi";
import { useArtifactRunner } from "../../../hooks/useArtifactRunner";
import { PanelErrorNotice } from "./mle/PanelErrorNotice";
import { type ReactNode } from "react";
import {
  getNormalizedBenchmarkArtifact,
  getRagAblation,
  getRagSourceRegistry,
  getRagTraceReplay,
  runLiveRagEval,
} from "../../../api/client";
import { MetricCard } from "../../../components/ui/MetricCard";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { LoadingPane, EmptyPane, ErrorPane } from "../../../components/ui/Spinner";
import type { AdminAnalytics, RagAblationResult } from "../../../types/api";
import { RefreshCw } from "lucide-react";
import { AiTrinityCard, ResearchPaperQueryTelemetryCard } from "./RagEvidenceCards";
import {
  RagAblationPanel,
  RagTraceReplayCard,
} from "./rag/RagSectionPanels";
import {
  asRecord,
  formatMaybeMs,
  formatMaybeNumber,
  formatMaybePercent,
  readPath,
  readString,
} from "./rag/ragArtifactFormatting";

interface Props { analytics?: AdminAnalytics }

export function RagSection({ analytics }: Props) {
  const { data: registry, status } = useApi(getRagSourceRegistry, []);
  const { data: ablation, status: ablationStatus } = useApi(getRagAblation, []);
  const { data: liveRag, status: liveRagStatus, refetch: refetchLiveRag } = useApi(
    () => getNormalizedBenchmarkArtifact("live_rag_eval"),
    [],
  );
  const { data: claimCitation, status: claimCitationStatus } = useApi(
    () => getNormalizedBenchmarkArtifact("claim_level_citation_eval"),
    [],
  );
  const { data: costLatency, status: costLatencyStatus } = useApi(
    () => getNormalizedBenchmarkArtifact("cost_latency_report"),
    [],
  );
  const { data: aiTrinity, status: aiTrinityStatus } = useApi(
    () => getNormalizedBenchmarkArtifact("ai_trinity_tradeoff"),
    [],
  );
  const { data: citationSelectorHoldout, status: citationSelectorHoldoutStatus } = useApi(
    () => getNormalizedBenchmarkArtifact("claim_conditioned_citation_selector_holdout"),
    [],
  );
  const { data: providerApiCapture, status: providerApiCaptureStatus } = useApi(
    () => getNormalizedBenchmarkArtifact("provider_api_path_capture"),
    [],
  );
  const { data: runtimeQuality, status: runtimeQualityStatus } = useApi(
    () => getNormalizedBenchmarkArtifact("runtime_quality_sentinel"),
    [],
  );
  const { data: retrievalGoldset, status: retrievalGoldsetStatus } = useApi(
    () => getNormalizedBenchmarkArtifact("retrieval_goldset_eval"),
    [],
  );
  const { data: ragBaselineComparison, status: ragBaselineComparisonStatus } = useApi(
    () => getNormalizedBenchmarkArtifact("rag_baseline_comparison"),
    [],
  );
  const { data: ragPairedComparison, status: ragPairedComparisonStatus } = useApi(
    () => getNormalizedBenchmarkArtifact("rag_paired_statistical_comparison"),
    [],
  );
  const { data: routeLatencyBudget, status: routeLatencyBudgetStatus } = useApi(
    () => getNormalizedBenchmarkArtifact("route_latency_budget"),
    [],
  );
  const { data: researchPaperTelemetry, status: researchPaperTelemetryStatus } = useApi(
    () => getNormalizedBenchmarkArtifact("research_paper_query_telemetry"),
    [],
  );
  const { data: traceReplay, status: traceReplayStatus } = useApi(() => getRagTraceReplay(8), []);
  // Previously a `useState` flag plus a `try/finally` with no `catch`, so a
  // failed rerun reset the spinner and then escaped as an unhandled promise
  // rejection with nothing shown to the operator.
  const {
    running: runningLiveRag,
    error: liveRagError,
    run: refreshLiveRag,
  } = useArtifactRunner(runLiveRagEval, refetchLiveRag, "admin.rag.liveRagEval");

  const rag = analytics?.rag_evaluation;

  return (
    <div className="flex flex-col gap-4">
      <PanelErrorNotice panel="Live RAG evaluation" error={liveRagError} />
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Evaluations" value={rag?.evaluations ?? null} />
        <MetricCard
          label="Grounding"
          value={rag?.grounding_score != null ? `${(rag.grounding_score * 100).toFixed(1)}%` : null}
          status={rag?.grounding_score != null && rag.grounding_score >= 0.8 ? "green" : "amber"}
        />
        <MetricCard
          label="Hallucination"
          value={rag?.hallucination_score != null ? `${(rag.hallucination_score * 100).toFixed(1)}%` : null}
          status={rag?.hallucination_score != null && rag.hallucination_score <= 0.05 ? "green" : "amber"}
        />
        <MetricCard
          label="Precision@3"
          value={rag?.precision_at_3 != null ? `${(rag.precision_at_3 * 100).toFixed(1)}%` : null}
        />
      </div>

      <ResearchPaperQueryTelemetryCard
        status={researchPaperTelemetryStatus}
        artifact={researchPaperTelemetry}
      />

      <AiTrinityCard status={aiTrinityStatus} artifact={aiTrinity} />

      <div className="grid lg:grid-cols-2 gap-4">
        <ArtifactSummaryCard
          title="Citation Selector Frozen Holdout"
          status={citationSelectorHoldoutStatus}
          artifact={citationSelectorHoldout}
          metrics={[
            ["Cases", ["metrics", "case_count"], "number"],
            ["Baseline citation", ["metrics", "baseline_citation_precision"], "percent"],
            ["Selector citation", ["metrics", "selector_citation_precision"], "percent"],
            ["Citation delta", ["metrics", "citation_precision_delta"], "percent"],
            ["Strict lift proven", ["metrics", "strict_improvement_proven"], "boolean"],
            ["Live route changed", ["metrics", "live_patient_route_changed"], "boolean"],
          ]}
          emptyLabel="No frozen selector holdout yet - run scripts/run_claim_conditioned_citation_selector_holdout.py"
        />
        <ArtifactSummaryCard
          title="Provider Usage: Normal API Path"
          status={providerApiCaptureStatus}
          artifact={providerApiCapture}
          metrics={[
            ["Requests", ["metrics", "request_count"], "number"],
            ["Usage coverage", ["metrics", "provider_usage_coverage_rate"], "percent"],
            ["Provider tokens", ["metrics", "provider_reported_total_tokens"], "number"],
            ["Estimated cost", ["metrics", "estimated_cost_usd"], "currency"],
            ["Completed", ["metrics", "completed"], "boolean"],
            ["Patient data", ["metrics", "patient_data_processed"], "boolean"],
          ]}
          emptyLabel="No provider-path probe artifact yet - run scripts/run_provider_api_path_capture.py"
        />
      </div>

      <Card>
        <CardHeader><SectionTitle>Cost & Latency</SectionTitle></CardHeader>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard label="Est. cost" value={rag?.estimated_cost_usd != null ? `$${rag.estimated_cost_usd.toFixed(4)}` : null} />
          <MetricCard label="Input tokens" value={rag?.input_tokens ?? null} />
          <MetricCard label="Output tokens" value={rag?.output_tokens ?? null} />
          <MetricCard
            label="P95 latency"
            value={rag?.p95_latency_ms != null ? `${rag.p95_latency_ms.toFixed(0)}ms` : null}
            status={rag?.p95_latency_ms != null && rag.p95_latency_ms < 3000 ? "green" : "amber"}
          />
        </div>
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle>RAG Ablation Study</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(139,92,246,0.12)", color: "#c4b5fd" }}>
            BM25 vs Sparse vs Dense Hybrid vs Reranked
          </span>
        </CardHeader>
        {ablationStatus === "loading" ? <LoadingPane /> :
         ablationStatus === "error" ? <ErrorPane message="Could not load RAG ablation" /> :
         !ablation ? <EmptyPane label="No ablation data - run POST /admin/rag-ablation first" /> : (
          <RagAblationPanel data={ablation as RagAblationResult} />
         )}
      </Card>

      <div className="grid lg:grid-cols-2 gap-4">
        <ArtifactSummaryCard
          title="Cost / Latency Observability"
          status={costLatencyStatus}
          artifact={costLatency}
          metrics={[
            ["Requests", ["metrics", "request_count"], "number"],
            ["P95 latency", ["metrics", "latency_p95_ms"], "milliseconds"],
            ["Local RAG P95", ["metrics", "normal_rag_probe_p95_ms"], "milliseconds"],
            ["Retrieval P95", ["metrics", "normal_rag_retrieval_p95_ms"], "milliseconds"],
            ["Provider tokens", ["metrics", "provider_reported_total_tokens"], "number"],
            ["Usage coverage", ["metrics", "provider_usage_coverage_rate"], "percent"],
            ["Est. pipeline tokens", ["metrics", "estimated_pipeline_total_tokens"], "number"],
            ["Est. cost", ["metrics", "estimated_total_cost_usd"], "currency"],
          ]}
          emptyLabel="No cost/latency report yet - run scripts/run_cost_latency_report.py"
        />
        <ArtifactSummaryCard
          title="Runtime Quality Sentinel"
          status={runtimeQualityStatus}
          artifact={runtimeQuality}
          metrics={[
            ["Alerts", ["metrics", "alert_count"], "number"],
            ["Unsafe answers", ["metrics", "unsafe_answer_rate"], "percent"],
            ["Unsupported claims", ["metrics", "unsupported_claim_rate"], "percent"],
            ["P95 latency", ["metrics", "latency_p95_ms"], "milliseconds"],
          ]}
          emptyLabel="No runtime quality snapshot yet - run scripts/run_runtime_quality_sentinel.py"
        />
        <ArtifactSummaryCard
          title="Retrieval Goldset"
          status={retrievalGoldsetStatus}
          artifact={retrievalGoldset}
          metrics={[
            ["Recall@10", ["metrics", "recall_at_10"], "percent"],
            ["MRR", ["metrics", "mrr"], "number"],
            ["Unsupported context", ["metrics", "unsupported_context_rate"], "percent"],
            ["Improvement proven", ["metrics", "improvement_proven"], "boolean"],
          ]}
          emptyLabel="No retrieval goldset eval yet - run scripts/run_retrieval_goldset_eval.py"
        />
        <RagBaselineComparisonCard status={ragBaselineComparisonStatus} artifact={ragBaselineComparison} />
        <ArtifactSummaryCard
          title="Paired RAG Evidence"
          status={ragPairedComparisonStatus}
          artifact={ragPairedComparison}
          metrics={[
            ["Paired cases", ["metrics", "goldset_case_count"], "number"],
            ["Recall@10 delta", ["metrics", "full_stack_recall_at_10_favorable_delta"], "percent"],
            ["Adjusted p", ["metrics", "full_stack_recall_at_10_adjusted_p_value"], "number"],
            ["Lift proven", ["metrics", "full_stack_improvement_proven_vs_bm25"], "boolean"],
          ]}
          emptyLabel="No paired comparison yet - run scripts/run_rag_paired_statistical_comparison.py"
        />
        <ArtifactSummaryCard
          title="Route Latency Budget"
          status={routeLatencyBudgetStatus}
          artifact={routeLatencyBudget}
          metrics={[
            ["Routes", ["metrics", "route_count"], "number"],
            ["Needs attention", ["metrics", "needs_attention_count"], "number"],
            ["Highest P95", ["metrics", "highest_observed_p95_ms"], "milliseconds"],
          ]}
          emptyLabel="No route latency budget yet - run scripts/run_route_latency_budget.py"
        />
        <ArtifactSummaryCard
          title="Live-Agent RAG Eval"
          status={liveRagStatus}
          artifact={liveRag}
          action={
            <Button
              variant="secondary"
              size="sm"
              loading={runningLiveRag}
              icon={<RefreshCw size={12} />}
              onClick={() => void refreshLiveRag()}
            >
              Rerun
            </Button>
          }
          metrics={[
            ["Pass rate", ["metrics", "pass_rate"], "percent"],
            ["Claims", ["metrics", "claim_support_rate"], "percent"],
            ["Tier correctness", ["metrics", "source_tier_correctness"], "percent"],
            ["Unsafe answers", ["metrics", "unsafe_answer_rate"], "percent"],
          ]}
          emptyLabel="No live RAG eval yet - run scripts/run_live_rag_eval.py"
        />
        <ArtifactSummaryCard
          title="Claim-Level Citation Eval"
          status={claimCitationStatus}
          artifact={claimCitation}
          metrics={[
            ["Cases", ["metrics", "case_count"], "number"],
            ["Hard failures", ["metrics", "hard_failures"], "number"],
            ["NLI-required", ["metrics", "nli_required_cases"], "number"],
            ["NLI available", ["metrics", "nli_available_cases"], "number"],
          ]}
          emptyLabel="No claim citation eval yet - run scripts/run_rag_claim_validation_eval.py"
        />
      </div>

      <Card>
        <CardHeader><SectionTitle>Knowledge Base Sources</SectionTitle></CardHeader>
        {status === "loading" && <LoadingPane />}
        {status === "success" && (
          <>
            {(registry?.sources ?? []).length === 0 ? (
              <EmptyPane label="No sources indexed" />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border)" }}>
                      {["Source", "Trust", "Chunks", "Topics"].map((h) => (
                        <th key={h} className="text-left py-2 pr-3 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {registry!.sources.map((src, index) => (
                      <tr key={`${src.id ?? src.source_name ?? "source"}-${index}`} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                        <td className="py-2 pr-3 font-medium max-w-[200px] truncate" style={{ color: "var(--text)" }}>{src.source_name}</td>
                        <td className="py-2 pr-3">
                          <Badge variant={src.trust_level === "high" ? "green" : src.trust_level === "medium" ? "amber" : "muted"}>
                            {src.trust_level}
                          </Badge>
                        </td>
                        <td className="py-2 pr-3 tabular-nums" style={{ color: "var(--text-dim)" }}>{src.chunk_count}</td>
                        <td className="py-2" style={{ color: "var(--text-faint)" }}>
                          {src.topics?.slice(0, 3).join(", ")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </Card>

      <RagTraceReplayCard status={traceReplayStatus} artifact={traceReplay} />
    </div>
  );
}

type MetricFormat = "number" | "percent" | "currency" | "milliseconds" | "boolean";

function ArtifactSummaryCard({
  title,
  status,
  artifact,
  action,
  metrics,
  emptyLabel,
}: {
  title: string;
  status: "idle" | "loading" | "success" | "error";
  artifact: unknown;
  action?: ReactNode;
  metrics: Array<[string, string[], MetricFormat]>;
  emptyLabel: string;
}) {
  const record = asRecord(artifact);
  const artifactStatus = readString(record, ["status"]);
  const claimBoundary = readString(record, ["claim_boundary"]);

  return (
    <Card>
      <CardHeader>
        <SectionTitle>{title}</SectionTitle>
        <div className="flex items-center gap-2">
          {artifactStatus && (
            <Badge variant={artifactStatus === "strong" || artifactStatus === "acceptable" ? "green" : "amber"}>
              {artifactStatus}
            </Badge>
          )}
          {action}
        </div>
      </CardHeader>
      {status === "loading" ? <LoadingPane /> :
       status === "error" ? <ErrorPane message={`Could not load ${title}`} /> :
       !record ? <EmptyPane label={emptyLabel} /> : (
        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-2 gap-3">
            {metrics.map(([label, path, format]) => (
              <MetricCard key={label} label={label} value={formatMetric(readPath(record, path), format)} />
            ))}
          </div>
          {claimBoundary && (
            <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{claimBoundary}</p>
          )}
          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Internal engineering eval. Clinical validation: false.
          </p>
        </div>
       )}
    </Card>
  );
}

function RagBaselineComparisonCard({
  status,
  artifact,
}: {
  status: "idle" | "loading" | "success" | "error";
  artifact: unknown;
}) {
  const record = asRecord(artifact);
  const rows = Array.isArray(record?.rows) ? record.rows.map(asRecord).filter(Boolean) as Record<string, unknown>[] : [];
  const artifactStatus = readString(record, ["status"]);
  const improvement = readPath(record, ["metrics", "improvement_proven_vs_bm25"]);
  const failures = rows
    .flatMap((row) => Array.isArray(row.failure_examples) ? row.failure_examples : [])
    .map(asRecord)
    .filter(Boolean)
    .slice(0, 4) as Record<string, unknown>[];

  return (
    <Card>
      <CardHeader>
        <SectionTitle>RAG Baseline Comparison</SectionTitle>
        <div className="flex items-center gap-2">
          <Badge variant={improvement === true ? "green" : "amber"}>
            {improvement === true ? "improvement proven" : "not proven"}
          </Badge>
          {artifactStatus && (
            <Badge variant={artifactStatus === "strong" || artifactStatus === "acceptable" ? "green" : "amber"}>
              {artifactStatus}
            </Badge>
          )}
        </div>
      </CardHeader>
      {status === "loading" ? <LoadingPane /> :
       status === "error" ? <ErrorPane message="Could not load RAG baseline comparison" /> :
       rows.length === 0 ? <EmptyPane label="No RAG baseline comparison yet - run scripts/run_rag_baseline_comparison.py" /> : (
        <div className="flex flex-col gap-3">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["Config", "Recall@10", "MRR", "Citation", "Unsupported", "Tier", "P95", "Failures"].map((h) => (
                    <th key={h} className="text-left py-2 pr-3 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={String(row.configuration ?? row.label)} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td className="py-2 pr-3 font-medium max-w-[190px] truncate" title={String(row.label ?? "")}>{String(row.label ?? row.configuration ?? "-")}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybePercent(row.recall_at_10)}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybeNumber(row.mrr)}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybePercent(row.citation_precision)}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybePercent(row.unsupported_context_rate)}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybePercent(row.source_tier_correctness)}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybeMs(row.latency_p95_ms)}</td>
                    <td className="py-2 pr-3 tabular-nums">{String(row.failure_count ?? "-")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {failures.length > 0 && (
            <div className="flex flex-col gap-2">
              <p className="text-xs font-semibold" style={{ color: "var(--text-dim)" }}>Failure examples</p>
              {failures.map((failure, index) => (
                <div key={`${String(failure.case_id ?? index)}-${index}`} className="rounded-md border p-2" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium" style={{ color: "var(--text)" }}>{String(failure.case_id ?? "case")}</span>
                    <span className="text-xs" style={{ color: "var(--text-faint)" }}>{Array.isArray(failure.failure_reasons) ? failure.failure_reasons.join(", ") : "-"}</span>
                  </div>
                  <p className="text-xs mt-1 line-clamp-2" style={{ color: "var(--text-dim)" }}>{String(failure.query ?? "")}</p>
                </div>
              ))}
            </div>
          )}

          <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>
            Internal frozen-goldset engineering comparison only. Clinical validation: false.
          </p>
        </div>
       )}
    </Card>
  );
}

function formatMetric(value: unknown, format: MetricFormat): string | null {
  if (format === "boolean") {
    if (typeof value !== "boolean") return null;
    return value ? "yes" : "no";
  }
  if (typeof value !== "number") return null;
  if (format === "percent") return `${(value * 100).toFixed(1)}%`;
  if (format === "currency") return `$${value.toFixed(6)}`;
  if (format === "milliseconds") return `${value.toFixed(0)}ms`;
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}
