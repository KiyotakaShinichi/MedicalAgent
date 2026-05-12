import { useApi } from "../../../hooks/useApi";
import { getRagSourceRegistry } from "../../../api/client";
import { MetricCard } from "../../../components/ui/MetricCard";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { Badge } from "../../../components/ui/Badge";
import { LoadingPane, EmptyPane } from "../../../components/ui/Spinner";
import type { AdminAnalytics } from "../../../types/api";

interface Props { analytics: AdminAnalytics }

export function RagSection({ analytics }: Props) {
  const { data: registry, status } = useApi(getRagSourceRegistry, []);
  const rag = analytics.rag_evaluation;

  return (
    <div className="flex flex-col gap-4">
      {/* RAG metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Evaluations"    value={rag.evaluations} />
        <MetricCard label="Grounding"      value={rag.grounding_score != null ? `${(rag.grounding_score * 100).toFixed(1)}%` : null}
          status={rag.grounding_score != null && rag.grounding_score >= 0.8 ? "green" : "amber"} />
        <MetricCard label="Hallucination"  value={rag.hallucination_score != null ? `${(rag.hallucination_score * 100).toFixed(1)}%` : null}
          status={rag.hallucination_score != null && rag.hallucination_score <= 0.05 ? "green" : "amber"} />
        <MetricCard label="Precision@3"    value={rag.precision_at_3 != null ? `${(rag.precision_at_3 * 100).toFixed(1)}%` : null} />
      </div>

      {/* Cost metrics */}
      <Card>
        <CardHeader><SectionTitle>Cost & Latency</SectionTitle></CardHeader>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard label="Est. cost"     value={rag.estimated_cost_usd != null ? `$${rag.estimated_cost_usd.toFixed(4)}` : null} />
          <MetricCard label="Input tokens"  value={rag.input_tokens} />
          <MetricCard label="Output tokens" value={rag.output_tokens} />
          <MetricCard label="P95 latency"   value={rag.p95_latency_ms != null ? `${rag.p95_latency_ms.toFixed(0)}ms` : null}
            status={rag.p95_latency_ms != null && rag.p95_latency_ms < 3000 ? "green" : "amber"} />
        </div>
      </Card>

      {/* KB sources */}
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
