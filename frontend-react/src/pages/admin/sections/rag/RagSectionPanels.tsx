import { Badge } from "../../../../components/ui/Badge";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { EmptyPane, ErrorPane, LoadingPane } from "../../../../components/ui/Spinner";
import type { AblationStrategyMetrics, RagAblationResult } from "../../../../types/api";
import { asRecord, readPath, readString } from "./ragArtifactFormatting";

export function RagTraceReplayCard({
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
                  {["When", "Intent", "Mode", "Answerability", "Confidence", "Reason", "Distress", "Refusal", "Claims", "Post-gen", "Cache", "Request", "Sources"].map((heading) => (
                    <th key={heading} className="text-left py-2 pr-3 font-medium" style={{ color: "var(--text-faint)" }}>{heading}</th>
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
      <AblationRow label="Cases" value={String(metrics.case_count)} />
      <AblationRow
        label="Pass rate"
        value={passRate != null ? `${(passRate * 100).toFixed(1)}%` : "-"}
        color={passRate != null && passRate >= 0.9 ? "var(--green)" : "var(--amber)"}
      />
      <AblationRow label="Source hit" value={hitRate != null ? `${(hitRate * 100).toFixed(1)}%` : String(metrics.expected_source_hit_rate ?? "n/a")} />
      <AblationRow label="Grounding" value={metrics.average_grounding_score != null ? metrics.average_grounding_score.toFixed(3) : "-"} />
      <AblationRow label="Avg latency" value={metrics.average_latency_ms != null ? `${metrics.average_latency_ms.toFixed(0)}ms` : "-"} />
      {metrics.backend && <AblationRow label="Backend" value={metrics.backend} />}
    </div>
  );
}

function AblationRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-xs" style={{ color: "var(--text-faint)" }}>{label}</span>
      <span className="text-xs tabular-nums font-medium text-right max-w-[170px] truncate" style={{ color: color ?? "var(--text-dim)" }}>{value}</span>
    </div>
  );
}

export function RagAblationPanel({ data }: { data: RagAblationResult }) {
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
        {strategies.map(([label, metrics], index) => (
          <div key={label}>
            <p className="text-xs font-semibold mb-1.5" style={{ color: "var(--text-dim)" }}>{label}</p>
            <AblationCell metrics={metrics} isWinner={index === winnerIndex} />
          </div>
        ))}
      </div>
      {data.comparison.notes.length > 0 && (
        <div className="flex flex-col gap-1">
          {data.comparison.notes.map((note, index) => (
            <p key={index} className="text-xs" style={{ color: "var(--text-dim)" }}>- {note}</p>
          ))}
        </div>
      )}
      <div className="flex flex-col gap-1 pt-1 border-t" style={{ borderColor: "var(--border)" }}>
        {data.limitations.map((limitation, index) => (
          <p key={index} className="text-xs" style={{ color: "var(--text-faint)" }}>Warning: {limitation}</p>
        ))}
        <p className="text-xs italic mt-1" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
      </div>
    </div>
  );
}
