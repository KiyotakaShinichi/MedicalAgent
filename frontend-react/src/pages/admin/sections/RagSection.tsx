import { useApi } from "../../../hooks/useApi";
import { useState, type ReactNode } from "react";
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
import type { AdminAnalytics, RagAblationResult, AblationStrategyMetrics } from "../../../types/api";
import { RefreshCw } from "lucide-react";

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
  const { data: traceReplay, status: traceReplayStatus } = useApi(() => getRagTraceReplay(8), []);
  const [runningLiveRag, setRunningLiveRag] = useState(false);
  const rag = analytics?.rag_evaluation;

  async function refreshLiveRag() {
    setRunningLiveRag(true);
    try {
      await runLiveRagEval();
      await refetchLiveRag();
    } finally {
      setRunningLiveRag(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
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
            ["Est. cost", ["metrics", "estimated_total_cost_usd"], "currency"],
            ["Cache hit", ["metrics", "cache_hit_rate"], "percent"],
          ]}
          emptyLabel="No cost/latency report yet - run scripts/run_cost_latency_report.py"
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

type MetricFormat = "number" | "percent" | "currency" | "milliseconds";

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
        </div>
       )}
    </Card>
  );
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function readPath(record: Record<string, unknown> | null, path: string[]): unknown {
  let current: unknown = record;
  for (const key of path) {
    if (!current || typeof current !== "object" || Array.isArray(current)) return null;
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

function readString(record: Record<string, unknown> | null, path: string[]): string | null {
  const value = readPath(record, path);
  return typeof value === "string" ? value : null;
}

function formatMetric(value: unknown, format: MetricFormat): string | null {
  if (typeof value !== "number") return null;
  if (format === "percent") return `${(value * 100).toFixed(1)}%`;
  if (format === "currency") return `$${value.toFixed(6)}`;
  if (format === "milliseconds") return `${value.toFixed(0)}ms`;
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

function RagTraceReplayCard({
  status,
  artifact,
}: {
  status: "idle" | "loading" | "success" | "error";
  artifact: unknown;
}) {
  const record = asRecord(artifact);
  const traces = Array.isArray(record?.traces) ? record.traces : [];
  return (
    <Card>
      <CardHeader>
        <SectionTitle>RAG Trace Replay</SectionTitle>
        <Badge variant={traces.length > 0 ? "green" : "muted"}>{traces.length} traces</Badge>
      </CardHeader>
      {status === "loading" ? <LoadingPane /> :
       status === "error" ? <ErrorPane message="Could not load RAG trace replay" /> :
       traces.length === 0 ? <EmptyPane label="No RAG trace rows recorded yet." /> : (
        <div className="flex flex-col gap-3">
          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Trace diagnostics apply to new RAG rows written after the trace-fields migration; older rows may show blank answerability or confidence fields.
          </p>
          <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["When", "Intent", "Mode", "Answerability", "Confidence", "Reason", "Distress", "Refusal", "Claims", "Post-gen", "Cache", "Request", "Sources"].map((h) => (
                  <th key={h} className="text-left py-2 pr-3 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {traces.slice(0, 8).map((trace, index) => {
                const row = asRecord(trace);
                const claim = asRecord(row?.claim_validation);
                const postGen = asRecord(row?.post_gen_validator);
                const retrieval = asRecord(row?.retrieval_confidence);
                const diagnostics = asRecord(row?.trace_diagnostics);
                const distress = asRecord(readPath(diagnostics, ["emotional_distress"]));
                const refusal = asRecord(readPath(diagnostics, ["refusal"]));
                const sources = Array.isArray(row?.retrieved_source_ids) ? row.retrieved_source_ids.length : 0;
                const confidence = readPath(retrieval, ["retrieval_confidence"]);
                return (
                  <tr key={String(row?.id ?? index)} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td className="py-2 pr-3" style={{ color: "var(--text-dim)" }}>{readString(row, ["created_at"]) ?? "—"}</td>
                    <td className="py-2 pr-3">{readString(row, ["intent"]) ?? "—"}</td>
                    <td className="py-2 pr-3">{readString(row, ["rag_mode"]) ?? "—"}</td>
                    <td className="py-2 pr-3">{String(readPath(retrieval, ["answerability_status"]) ?? "-")}</td>
                    <td className="py-2 pr-3 tabular-nums">{typeof confidence === "number" ? `${(confidence * 100).toFixed(0)}%` : "-"}</td>
                    <td className="py-2 pr-3 max-w-[220px] truncate" title={String(readPath(retrieval, ["reason"]) ?? "")}>{String(readPath(retrieval, ["reason"]) ?? "-")}</td>
                    <td className="py-2 pr-3">{String(readPath(distress, ["response_mode"]) ?? "-")}</td>
                    <td className="py-2 pr-3 max-w-[180px] truncate" title={String(readPath(refusal, ["refusal_reason"]) ?? "")}>{String(readPath(refusal, ["refusal_reason"]) ?? "-")}</td>
                    <td className="py-2 pr-3">{String(readPath(claim, ["citation_status"]) ?? "—")}</td>
                    <td className="py-2 pr-3">{String(readPath(postGen, ["decision"]) ?? "—")}</td>
                    <td className="py-2 pr-3">{readString(row, ["cache_status"]) ?? "-"}</td>
                    <td className="py-2 pr-3 max-w-[160px] truncate" title={String(readPath(diagnostics, ["correlation_id"]) ?? "")}>{String(readPath(diagnostics, ["correlation_id"]) ?? "-")}</td>
                    <td className="py-2 pr-3 tabular-nums">{sources}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          </div>
        </div>
       )}
    </Card>
  );
}

function AblationCell({ metrics, isWinner }: { metrics: AblationStrategyMetrics; isWinner: boolean }) {
  const passRate = typeof metrics.pass_rate === "number" ? metrics.pass_rate : null;
  const hitRate = typeof metrics.expected_source_hit_rate === "number" ? metrics.expected_source_hit_rate : null;
  return (
    <div
      className="rounded-md border p-3 flex flex-col gap-2"
      style={{
        background: isWinner ? "rgba(16,185,129,0.06)" : "var(--surface)",
        borderColor: isWinner ? "rgba(16,185,129,0.35)" : "var(--border)",
      }}
    >
      {isWinner && <span className="text-xs font-semibold" style={{ color: "var(--green)" }}>winner</span>}
      <AblRow label="Cases" value={String(metrics.case_count)} />
      <AblRow
        label="Pass rate"
        value={passRate != null ? `${(passRate * 100).toFixed(1)}%` : "-"}
        color={passRate != null && passRate >= 0.9 ? "var(--green)" : "var(--amber)"}
      />
      <AblRow label="Source hit" value={hitRate != null ? `${(hitRate * 100).toFixed(1)}%` : String(metrics.expected_source_hit_rate ?? "n/a")} />
      <AblRow label="Grounding" value={metrics.average_grounding_score != null ? metrics.average_grounding_score.toFixed(3) : "-"} />
      <AblRow label="Avg latency" value={metrics.average_latency_ms != null ? `${metrics.average_latency_ms.toFixed(0)}ms` : "-"} />
      {metrics.backend && <AblRow label="Backend" value={metrics.backend} />}
    </div>
  );
}

function AblRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-xs" style={{ color: "var(--text-faint)" }}>{label}</span>
      <span className="text-xs tabular-nums font-medium text-right max-w-[170px] truncate" style={{ color: color ?? "var(--text-dim)" }}>{value}</span>
    </div>
  );
}

function RagAblationPanel({ data }: { data: RagAblationResult }) {
  const strategyCandidates: Array<[string, AblationStrategyMetrics | undefined]> = [
    ["BM25 only", data.strategies.bm25_only],
    ["Sparse BM25 + TF-IDF", data.strategies.sparse_tfidf_bm25],
    ["Dense FAISS + BM25 + RRF", data.strategies.dense_faiss_bm25_rrf ?? data.strategies.hybrid],
    ["Agent-boosted hybrid", data.strategies.dense_faiss_bm25_rrf_agent_boosted],
    ["Full reranked pipeline", data.strategies.dense_faiss_bm25_rrf_reranked ?? data.strategies.hybrid_reranked],
  ];
  const strategies: Array<[string, AblationStrategyMetrics]> = strategyCandidates.filter(
    (entry): entry is [string, AblationStrategyMetrics] => Boolean(entry[1]),
  );

  const winnerIndex = Math.max(0, strategies.findIndex(([label]) => label === "Full reranked pipeline"));

  return (
    <div className="flex flex-col gap-3">
      {data.active_index && (
        <div className="rounded-md border p-3 text-xs" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
          <p style={{ color: "var(--text)" }}>
            Active backend: <strong>{data.active_index.retrieval_backend ?? "unknown"}</strong>
          </p>
          <p style={{ color: "var(--text-faint)" }}>
            Dense: {data.active_index.dense_component ?? "unavailable"} | Sparse: {data.active_index.sparse_component ?? "unknown"} | Fusion: {data.active_index.fusion ?? "n/a"}
          </p>
        </div>
      )}

      <div className="grid sm:grid-cols-2 xl:grid-cols-5 gap-3">
        {strategies.map(([label, metrics], i) => (
          <div key={label}>
            <p className="text-xs font-semibold mb-1.5" style={{ color: "var(--text-dim)" }}>{label}</p>
            <AblationCell metrics={metrics} isWinner={i === winnerIndex} />
          </div>
        ))}
      </div>

      {data.comparison.notes.length > 0 && (
        <div className="flex flex-col gap-1">
          {data.comparison.notes.map((note, i) => (
            <p key={i} className="text-xs" style={{ color: "var(--text-dim)" }}>- {note}</p>
          ))}
        </div>
      )}

      <div className="flex flex-col gap-1 pt-1 border-t" style={{ borderColor: "var(--border)" }}>
        {data.limitations.map((lim, i) => (
          <p key={i} className="text-xs" style={{ color: "var(--text-faint)" }}>Warning: {lim}</p>
        ))}
        <p className="text-xs italic mt-1" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
      </div>
    </div>
  );
}
