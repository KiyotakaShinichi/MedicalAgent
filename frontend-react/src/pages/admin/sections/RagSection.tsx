import { useApi } from "../../../hooks/useApi";
import { getRagSourceRegistry, getRagAblation } from "../../../api/client";
import { MetricCard } from "../../../components/ui/MetricCard";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { Badge } from "../../../components/ui/Badge";
import { LoadingPane, EmptyPane, ErrorPane } from "../../../components/ui/Spinner";
import type { AdminAnalytics, RagAblationResult, AblationStrategyMetrics } from "../../../types/api";

interface Props { analytics: AdminAnalytics }

export function RagSection({ analytics }: Props) {
  const { data: registry, status } = useApi(getRagSourceRegistry, []);
  const { data: ablation, status: ablationStatus } = useApi(getRagAblation, []);
  const rag = analytics.rag_evaluation;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Evaluations" value={rag.evaluations} />
        <MetricCard
          label="Grounding"
          value={rag.grounding_score != null ? `${(rag.grounding_score * 100).toFixed(1)}%` : null}
          status={rag.grounding_score != null && rag.grounding_score >= 0.8 ? "green" : "amber"}
        />
        <MetricCard
          label="Hallucination"
          value={rag.hallucination_score != null ? `${(rag.hallucination_score * 100).toFixed(1)}%` : null}
          status={rag.hallucination_score != null && rag.hallucination_score <= 0.05 ? "green" : "amber"}
        />
        <MetricCard
          label="Precision@3"
          value={rag.precision_at_3 != null ? `${(rag.precision_at_3 * 100).toFixed(1)}%` : null}
        />
      </div>

      <Card>
        <CardHeader><SectionTitle>Cost & Latency</SectionTitle></CardHeader>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard label="Est. cost" value={rag.estimated_cost_usd != null ? `$${rag.estimated_cost_usd.toFixed(4)}` : null} />
          <MetricCard label="Input tokens" value={rag.input_tokens} />
          <MetricCard label="Output tokens" value={rag.output_tokens} />
          <MetricCard
            label="P95 latency"
            value={rag.p95_latency_ms != null ? `${rag.p95_latency_ms.toFixed(0)}ms` : null}
            status={rag.p95_latency_ms != null && rag.p95_latency_ms < 3000 ? "green" : "amber"}
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
                    {registry!.sources.map((src) => (
                      <tr key={src.id} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
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
    </div>
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
