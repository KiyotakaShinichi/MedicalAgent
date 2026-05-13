import { useState } from "react";
import { ChevronDown, ChevronRight, Clock, Database, Shield, Zap } from "lucide-react";
import { useApi } from "../../../hooks/useApi";
import { getAdminAnalytics, getAgentTraceLogs } from "../../../api/client";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { Badge } from "../../../components/ui/Badge";
import { statusVariant } from "../../../components/ui/badgeUtils";
import { MetricCard } from "../../../components/ui/MetricCard";
import { LoadingPane, ErrorPane, EmptyPane } from "../../../components/ui/Spinner";
import type { AgentTraceLog } from "../../../types/api";

type TraceEntry = AgentTraceLog;

function TraceRow({ trace, index }: { trace: TraceEntry; index: number }) {
  const [open, setOpen] = useState(false);

  const intentColor = (intent: string) => {
    if (intent.includes("security") || intent.includes("safety")) return "red";
    if (intent.includes("education") || intent.includes("portal")) return "blue";
    if (intent.includes("conversation") || intent.includes("emotional")) return "purple";
    if (intent.includes("data_entry") || intent.includes("timeline")) return "cyan";
    return "muted";
  };

  return (
    <div style={{ borderBottom: "1px solid var(--border)" }}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:opacity-80 transition-opacity"
      >
        <span style={{ color: "var(--text-faint)", flexShrink: 0, fontSize: 11, width: 20 }}>
          {index + 1}
        </span>
        {open ? <ChevronDown size={12} style={{ flexShrink: 0, color: "var(--text-faint)" }} />
               : <ChevronRight size={12} style={{ flexShrink: 0, color: "var(--text-faint)" }} />}

        <span className="flex-1 min-w-0 text-xs truncate" style={{ color: "var(--text)" }}>
          {trace.query_preview}
        </span>

        <div className="flex items-center gap-2 flex-shrink-0">
          <Badge variant={intentColor(trace.intent ?? "") as "red" | "blue" | "purple" | "cyan" | "muted"}>
            {(trace.intent ?? "unknown").replace(/_/g, " ")}
          </Badge>
          <Badge variant={statusVariant(trace.safety_level ?? "")}>
            {trace.safety_level ?? "-"}
          </Badge>
          <Badge variant={trace.input_guardrail === "passed" ? "green" : "red"}>
            {trace.input_guardrail === "passed" ? "pass" : "block"}
          </Badge>
          {trace.latency_ms != null && (
            <span className="text-xs tabular-nums" style={{ color: "var(--text-faint)", minWidth: 52, textAlign: "right" }}>
              {trace.latency_ms.toFixed(0)}ms
            </span>
          )}
        </div>
      </button>

      {open && (
        <div className="px-10 pb-3 grid gap-2" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
          <TraceDetail
            label="Route / Intent"
            icon={<Zap size={11} style={{ color: "var(--cyan)" }} />}
            value={trace.intent ?? "-"}
            sub={`Safety: ${trace.safety_level ?? "-"} / Input: ${trace.input_guardrail ?? "-"}`}
          />
          <TraceDetail
            label="Cache"
            icon={<Database size={11} style={{ color: "var(--blue)" }} />}
            value={trace.cache_status ?? "-"}
            sub={trace.cache_status === "hit" ? "Served from cache" : "Fresh generation"}
          />
          <TraceDetail
            label="RAG Sources"
            icon={<Database size={11} style={{ color: "var(--purple)" }} />}
            value={`${trace.retrieved_source_ids?.length ?? 0} chunks`}
            sub={trace.retrieved_source_ids?.slice(0, 3).join(", ") || "-"}
          />
          <TraceDetail
            label="Safety gate"
            icon={<Shield size={11} style={{ color: "var(--green)" }} />}
            value={trace.input_guardrail ?? "-"}
            sub={`Grounding: ${trace.grounding_score != null ? (trace.grounding_score * 100).toFixed(0) + "%" : "-"} / Hallucination risk: ${trace.hallucination_score != null ? (trace.hallucination_score * 100).toFixed(0) + "%" : "-"}`}
          />
          <TraceDetail
            label="Latency / tokens"
            icon={<Clock size={11} style={{ color: "var(--amber)" }} />}
            value={trace.latency_ms != null ? `${trace.latency_ms.toFixed(0)} ms` : "-"}
            sub={`~${trace.estimated_total_tokens?.toFixed(0) ?? "-"} tokens`}
          />
          {(trace.cited_source_ids?.length ?? 0) > 0 && (
            <div className="col-span-2 text-xs" style={{ color: "var(--text-faint)" }}>
              <span style={{ color: "var(--text-dim)" }}>Citations: </span>
              {trace.cited_source_ids.join(", ")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TraceDetail({ label, icon, value, sub }: { label: string; icon: React.ReactNode; value: string; sub?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center gap-1">
        {icon}
        <span className="text-xs" style={{ color: "var(--text-faint)" }}>{label}</span>
      </div>
      <p className="text-xs font-medium" style={{ color: "var(--text)" }}>{value}</p>
      {sub && <p className="text-xs" style={{ color: "var(--text-faint)" }}>{sub}</p>}
    </div>
  );
}

export function AgentTraceSection() {
  const { data: analytics, status } = useApi(getAdminAnalytics, []);

  const rag = analytics?.rag_evaluation;

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader><SectionTitle>Agent Pipeline Architecture</SectionTitle></CardHeader>
        <div className="grid gap-3 sm:grid-cols-2">
          <PipelineStep step="1" label="Input Safety Gate" color="var(--rose)"
            items={[
              "Prompt injection detection (EN + multilingual)",
              "PHI boundary: own patient only",
              "Multilingual attack pattern matching",
              "Base64 / encoded payload detection",
            ]} />
          <PipelineStep step="2" label="Intent Router" color="var(--blue)"
            items={[
              "Deterministic priority rules for obvious safety and tool requests",
              "LLM adjudication fallback for ambiguous queries",
              "Routes: security, safety, treatment boundary, education, portal help, conversation, emotional support, data entry",
            ]} />
          <PipelineStep step="3" label="RAG Retrieval (if triggered)" color="var(--purple)"
            items={[
              "Dense sentence-transformer retrieval with FAISS when available",
              "BM25 sparse retrieval fused with dense scores using RRF",
              "Sparse BM25 + TF-IDF fallback with honest backend labels",
              "Curated-source boost, parent-child window expansion, reranking, contextual compression",
              "Safety-aware semantic cache gate at similarity 0.86",
            ]} />
          <PipelineStep step="4" label="Output Validation" color="var(--green)"
            items={[
              "Grounding score vs retrieved context",
              "Hallucination risk flag",
              "Citation attachment",
              "Audit log write for latency, tokens, route, and sources",
            ]} />
        </div>
      </Card>

      {status === "loading" && <LoadingPane />}
      {rag && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard label="Avg grounding" value={rag.grounding_score != null ? `${(rag.grounding_score * 100).toFixed(1)}%` : null}
            status={rag.grounding_score != null && rag.grounding_score >= 0.8 ? "green" : "amber"} />
          <MetricCard label="Hallucination risk" value={rag.hallucination_score != null ? `${(rag.hallucination_score * 100).toFixed(1)}%` : null}
            status={rag.hallucination_score != null && rag.hallucination_score <= 0.08 ? "green" : "amber"} />
          <MetricCard label="Cache hit rate" value={rag.cache_hit_rate != null ? `${(rag.cache_hit_rate * 100).toFixed(0)}%` : null} />
          <MetricCard label="P95 latency" value={rag.p95_latency_ms != null ? `${rag.p95_latency_ms.toFixed(0)}ms` : null}
            status={rag.p95_latency_ms != null && rag.p95_latency_ms < 5000 ? "green" : "amber"} />
        </div>
      )}

      <Card>
        <CardHeader><SectionTitle>Per-Call Trace Log</SectionTitle></CardHeader>
        <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
          Each row below represents one agent invocation from regression evals or live usage. Expand a row to inspect route,
          intent, safety decision, cache status, RAG sources, grounding, latency, and token estimate.
        </p>
        <TraceLogFromLatestEval />
      </Card>
    </div>
  );
}

function TraceLogFromLatestEval() {
  const { data, status } = useApi(getAgentTraceLogs, []);

  if (status === "loading") return <LoadingPane />;
  if (status === "error") return <ErrorPane message="Could not load trace logs from backend" />;

  const traces = data?.traces ?? [];

  if (traces.length === 0) {
    return <EmptyPane label="No trace logs yet - run the agent regression suite or send a chat message to generate logs" />;
  }

  return (
    <div>
      <p className="text-xs mb-2 px-1" style={{ color: "var(--text-faint)" }}>
        Showing {traces.length} most recent agent invocations from the live DB. Each row is one full pipeline call:
        input gate -&gt; intent router -&gt; RAG -&gt; output gate.
      </p>
      <div className="rounded-md border overflow-hidden" style={{ borderColor: "var(--border)" }}>
        {traces.map((t, i) => <TraceRow key={t.id} trace={t} index={i} />)}
      </div>
    </div>
  );
}

function PipelineStep({ step, label, color, items }: {
  step: string; label: string; color: string; items: string[];
}) {
  return (
    <div className="rounded-md border p-3" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
      <div className="flex items-center gap-2 mb-2">
        <span
          className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
          style={{ background: color, color: "#fff" }}
        >
          {step}
        </span>
        <p className="text-xs font-semibold" style={{ color: "var(--text)" }}>{label}</p>
      </div>
      <ul className="flex flex-col gap-1">
        {items.map((item, i) => (
          <li key={i} className="text-xs flex items-start gap-1.5" style={{ color: "var(--text-dim)" }}>
            <span style={{ color: "var(--text-faint)", flexShrink: 0, marginTop: 2 }}>-</span>
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
