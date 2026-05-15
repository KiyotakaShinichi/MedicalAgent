import { useEffect, useMemo, useState, useCallback } from "react";
import { Wrench, CheckCircle2, XCircle, Clock, AlertCircle } from "lucide-react";
import { SectionCard } from "../../../components/ui/SectionCard";
import { StatusBadge } from "../../../components/ui/StatusBadge";
import { ErrorPane, LoadingPane } from "../../../components/ui/Spinner";

interface ToolActionCase {
  id: string;
  status: "passed" | "failed";
  latency_ms?: number;
  action_types?: string[];
  expected_actions?: string[];
  observed_actions?: string[];
  reply_preview?: string;
  error?: string;
}

interface ToolActionBenchmark {
  schema_version?: string;
  generated_at?: string;
  status?: "passed" | "failed" | "needs_attention" | string;
  summary?: {
    case_count?: number;
    passed?: number;
    pass_rate?: number;
    max_latency_ms?: number;
    average_latency_ms?: number;
  };
  cases?: ToolActionCase[];
  claim_boundary?: string;
}

const ARTIFACT_URL = "/artifacts/evals/tool_actions/latest_tool_action_benchmark.json";

function tone(status?: string): "success" | "danger" | "warning" | "neutral" {
  if (status === "passed") return "success";
  if (status === "failed") return "danger";
  if (status === "needs_attention") return "warning";
  return "neutral";
}

export function ToolActionBenchmarkSection() {
  const [data, setData] = useState<ToolActionBenchmark | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchBenchmark = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch(ARTIFACT_URL, { cache: "no-cache" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}: ${r.statusText}`);
        return r.json();
      })
      .then((json: ToolActionBenchmark) => {
        setData(json);
        setLoading(false);
      })
      .catch((e: Error) => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(fetchBenchmark, 0);
    return () => window.clearTimeout(timer);
  }, [fetchBenchmark]);

  const generatedAt = data?.generated_at;
  const generatedLabel = useMemo(() => {
    if (!generatedAt) return undefined;
    try {
      return new Date(generatedAt).toLocaleString();
    } catch { return generatedAt; }
  }, [generatedAt]);

  if (loading) return <LoadingPane label="Loading tool action benchmark..." />;
  if (error || !data) {
    return (
      <ErrorPane
        message={error ?? "Tool action benchmark artifact not found at /artifacts/evals/tool_actions/."}
        onRetry={fetchBenchmark}
      />
    );
  }

  const summary = data.summary ?? {};
  const cases = data.cases ?? [];
  const passRate = summary.pass_rate;
  const passRatePct = passRate != null ? `${(passRate * 100).toFixed(0)}%` : "—";

  return (
    <SectionCard
      title="Tool action benchmark"
      icon={Wrench}
      meta={generatedLabel ? `Generated ${generatedLabel}` : undefined}
    >
      {/* Headline metrics */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
          gap: 12,
          marginBottom: 16,
        }}
      >
        <Metric
          label="Status"
          value={
            <StatusBadge tone={tone(data.status)} size="md">
              {data.status ?? "unknown"}
            </StatusBadge>
          }
        />
        <Metric label="Pass rate"   value={<strong style={{ fontSize: "1.4rem", fontVariantNumeric: "tabular-nums" }}>{passRatePct}</strong>} sub={`${summary.passed ?? 0} / ${summary.case_count ?? cases.length}`} />
        <Metric label="Avg latency" value={<strong style={{ fontSize: "1.4rem", fontVariantNumeric: "tabular-nums" }}>{summary.average_latency_ms != null ? `${Math.round(summary.average_latency_ms)} ms` : "—"}</strong>} />
        <Metric label="Max latency" value={<strong style={{ fontSize: "1.4rem", fontVariantNumeric: "tabular-nums" }}>{summary.max_latency_ms != null ? `${Math.round(summary.max_latency_ms)} ms` : "—"}</strong>} />
      </div>

      {/* Per-case rows */}
      <div style={{ borderTop: "1px solid var(--border)", paddingTop: 10 }}>
        <p
          className="text-[0.72rem] font-semibold uppercase tracking-wider mb-2"
          style={{ color: "var(--text-faint)" }}
        >
          Cases
        </p>
        <ul style={{ display: "flex", flexDirection: "column", gap: 8, listStyle: "none", padding: 0, margin: 0 }}>
          {cases.map((c) => (
            <li
              key={c.id}
              style={{
                display: "grid",
                gridTemplateColumns: "auto minmax(0, 1fr) auto auto",
                gap: 10,
                alignItems: "center",
                padding: "10px 12px",
                border: "1px solid var(--border)",
                borderRadius: 8,
                background: "var(--surface)",
              }}
            >
              {c.status === "passed" ? (
                <CheckCircle2 size={16} style={{ color: "#047857", flexShrink: 0 }} aria-hidden="true" />
              ) : (
                <XCircle size={16} style={{ color: "#b91c1c", flexShrink: 0 }} aria-hidden="true" />
              )}
              <div style={{ minWidth: 0 }}>
                <p style={{ margin: 0, fontFamily: "var(--mono)", fontSize: "0.8rem", color: "var(--text-strong)" }}>
                  {c.id}
                </p>
                {c.action_types && c.action_types.length > 0 && (
                  <p style={{ margin: "2px 0 0", fontSize: "0.74rem", color: "var(--text-dim)" }}>
                    {c.action_types.join(" · ")}
                  </p>
                )}
                {c.error && (
                  <p style={{ margin: "2px 0 0", fontSize: "0.72rem", color: "#b91c1c", display: "flex", alignItems: "center", gap: 4 }}>
                    <AlertCircle size={11} /> {c.error}
                  </p>
                )}
              </div>
              {c.latency_ms != null && (
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 4,
                    color: "var(--text-faint)",
                    fontSize: "0.74rem",
                    fontVariantNumeric: "tabular-nums",
                    whiteSpace: "nowrap",
                  }}
                >
                  <Clock size={11} /> {Math.round(c.latency_ms)} ms
                </span>
              )}
              <StatusBadge tone={c.status === "passed" ? "success" : "danger"} size="sm">
                {c.status}
              </StatusBadge>
            </li>
          ))}
        </ul>
      </div>

      {data.claim_boundary && (
        <p
          className="text-[0.74rem] mt-4"
          style={{ color: "var(--text-faint)", lineHeight: 1.5 }}
        >
          {data.claim_boundary}
        </p>
      )}
    </SectionCard>
  );
}

function Metric({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div style={{ padding: "10px 12px", border: "1px solid var(--border)", borderRadius: 10, background: "var(--surface2)" }}>
      <p
        style={{
          margin: 0,
          fontSize: "0.7rem",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          color: "var(--text-faint)",
          fontWeight: 600,
        }}
      >
        {label}
      </p>
      <div style={{ marginTop: 4, color: "var(--text-strong)" }}>{value}</div>
      {sub && (
        <p style={{ margin: "2px 0 0", fontSize: "0.74rem", color: "var(--text-faint)" }}>{sub}</p>
      )}
    </div>
  );
}
