import { Badge } from "../../../components/ui/Badge";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { MetricCard } from "../../../components/ui/MetricCard";
import { EmptyPane, ErrorPane, LoadingPane } from "../../../components/ui/Spinner";

type ArtifactStatus = "idle" | "loading" | "success" | "error";
type MetricFormat = "number" | "percent" | "currency" | "milliseconds" | "boolean";

export function ResearchPaperQueryTelemetryCard({
  status,
  artifact,
}: {
  status: ArtifactStatus;
  artifact: unknown;
}) {
  const record = asRecord(artifact);
  const rows = Array.isArray(record?.rows)
    ? record.rows.map(asRecord).filter(Boolean) as Record<string, unknown>[]
    : [];
  const metrics = asRecord(record?.metrics);
  const artifactStatus = readString(record, ["status"]);
  const providerCoverage = metrics?.provider_usage_coverage_rate;

  return (
    <Card>
      <CardHeader>
        <SectionTitle>Research-Paper Query Telemetry</SectionTitle>
        <div className="flex items-center gap-2">
          <Badge variant={providerCoverage === 1 ? "green" : "amber"}>
            {providerCoverage === 1 ? "provider usage" : "token estimates"}
          </Badge>
          {artifactStatus && <Badge variant={artifactStatus === "acceptable_internal_measurement" ? "green" : "amber"}>{artifactStatus}</Badge>}
        </div>
      </CardHeader>
      {status === "loading" ? <LoadingPane /> :
       status === "error" ? <ErrorPane message="Could not load research-paper query telemetry" /> :
       rows.length === 0 ? <EmptyPane label="No query telemetry yet - run scripts/run_research_paper_query_telemetry.py" /> : (
        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard label="Queries" value={formatMetric(metrics?.query_count, "number")} />
            <MetricCard label="Route pass" value={formatMetric(metrics?.route_contract_pass_rate, "percent")} />
            <MetricCard label="Latency P50" value={formatMetric(metrics?.latency_p50_ms, "milliseconds")} />
            <MetricCard label="Latency P95" value={formatMetric(metrics?.latency_p95_ms, "milliseconds")} />
            <MetricCard label="Cold start" value={formatMetric(metrics?.cold_start_latency_ms, "milliseconds")} />
            <MetricCard label="Warm P95" value={formatMetric(metrics?.warm_latency_p95_ms, "milliseconds")} />
            <MetricCard label="Est. tokens" value={formatMetric(metrics?.estimated_pipeline_total_tokens, "number")} />
            <MetricCard label="Provider tokens" value={formatMetric(metrics?.provider_reported_total_tokens, "number")} />
            <MetricCard label="Usage coverage" value={formatMetric(metrics?.provider_usage_coverage_rate, "percent")} />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["Query", "Category", "Intent", "Route", "Latency", "Est. tokens", "Provider tokens", "Basis"].map((heading) => (
                    <th key={heading} className="text-left py-2 pr-3 font-medium" style={{ color: "var(--text-faint)" }}>{heading}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 12).map((row) => (
                  <tr key={String(row.id)} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td className="py-2 pr-3 max-w-[320px] truncate" title={String(row.query ?? "")}>{String(row.query ?? "-")}</td>
                    <td className="py-2 pr-3">{String(row.category ?? "-")}</td>
                    <td className="py-2 pr-3">{String(row.observed_intent ?? "-")}</td>
                    <td className="py-2 pr-3"><Badge variant={row.route_matches_contract === true ? "green" : "amber"}>{row.route_matches_contract === true ? "pass" : "review"}</Badge></td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybeMs(row.latency_ms)}</td>
                    <td className="py-2 pr-3 tabular-nums">{String(row.estimated_total_tokens ?? "-")}</td>
                    <td className="py-2 pr-3 tabular-nums">{String(row.provider_reported_total_tokens ?? "-")}</td>
                    <td className="py-2 pr-3">{String(row.token_measurement_basis ?? "-")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Fixed internal synthetic queries. Provider tokens appear only when the provider returns usage; otherwise values are clearly labeled estimates. Clinical validation: false.
          </p>
        </div>
       )}
    </Card>
  );
}

export function AiTrinityCard({
  status,
  artifact,
}: {
  status: ArtifactStatus;
  artifact: unknown;
}) {
  const record = asRecord(artifact);
  const metrics = asRecord(record?.metrics);
  const rows = Array.isArray(record?.rows)
    ? record.rows.map(asRecord).filter(Boolean) as Record<string, unknown>[]
    : [];
  const decision = typeof metrics?.decision === "string" ? metrics.decision : null;
  const costStatus = typeof metrics?.unit_cost_status === "string" ? metrics.unit_cost_status : null;

  return (
    <Card>
      <CardHeader>
        <SectionTitle>AI Trinity: Accuracy, Latency, Unit Cost</SectionTitle>
        <div className="flex items-center gap-2">
          <Badge variant={metrics?.promotion_allowed === true ? "green" : "amber"}>
            {metrics?.promotion_allowed === true ? "promotion allowed" : "hold"}
          </Badge>
          {decision && <Badge variant="muted">{decision}</Badge>}
        </div>
      </CardHeader>
      {status === "loading" ? <LoadingPane /> :
       status === "error" ? <ErrorPane message="Could not load AI Trinity evidence" /> :
       !record ? <EmptyPane label="No AI Trinity artifact yet - run scripts/run_ai_trinity_tradeoff.py" /> : (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard
              label="Accuracy / grounding"
              value={formatMetric(metrics?.accuracy_grounding_score, "percent")}
              status={metrics?.accuracy_status === "pass" ? "green" : "amber"}
            />
            <MetricCard
              label="Retrieval P95"
              value={formatMetric(metrics?.retrieval_p95_ms, "milliseconds")}
              status={metrics?.latency_status === "pass" ? "green" : "amber"}
            />
            <MetricCard
              label="Unit cost / safe answer"
              value={formatMetric(metrics?.cost_per_safe_supported_answer_usd, "currency") ?? "Not measured"}
              status={costStatus === "pass" ? "green" : "amber"}
            />
            <MetricCard
              label="Provider usage coverage"
              value={formatMetric(metrics?.provider_usage_coverage_rate, "percent")}
              status={metrics?.provider_usage_coverage_rate === 1 ? "green" : "amber"}
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["Configuration", "Quality", "Recall@10", "Citation", "Tier", "P95", "Compute index", "Decision"].map((heading) => (
                    <th key={heading} className="text-left py-2 pr-3 font-medium" style={{ color: "var(--text-faint)" }}>{heading}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={String(row.configuration)} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td className="py-2 pr-3 max-w-[260px] truncate" title={String(row.configuration ?? "")}>{String(row.configuration ?? "-")}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybePercent(row.accuracy_grounding_score)}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybePercent(row.recall_at_10)}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybePercent(row.citation_precision)}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybePercent(row.source_tier_correctness)}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybeMs(row.latency_p95_ms)}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatMaybeNumber(row.relative_local_compute_index)}</td>
                    <td className="py-2 pr-3">
                      <Badge variant={row.promotion_eligible === true ? "green" : "amber"}>
                        {row.promotion_eligible === true ? "eligible" : "blocked"}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Safety and grounding floors are binding. Missing provider telemetry is unknown, never $0. Internal engineering evidence only; clinical validation and production SLO: false.
          </p>
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
    current = asRecord(current)?.[key];
  }
  return current;
}

function readString(record: Record<string, unknown> | null, path: string[]): string | null {
  const value = readPath(record, path);
  return typeof value === "string" ? value : null;
}

function formatMaybePercent(value: unknown): string {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "-";
}

function formatMaybeNumber(value: unknown): string {
  return typeof value === "number" ? value.toFixed(2) : "-";
}

function formatMaybeMs(value: unknown): string {
  return typeof value === "number" ? `${value.toFixed(0)}ms` : "-";
}

function formatMetric(value: unknown, format: MetricFormat): string | null {
  if (format === "boolean") return typeof value === "boolean" ? (value ? "Yes" : "No") : null;
  if (typeof value !== "number") return null;
  if (format === "percent") return `${(value * 100).toFixed(1)}%`;
  if (format === "currency") return `$${value.toFixed(6)}`;
  if (format === "milliseconds") return `${value.toFixed(0)}ms`;
  return value.toLocaleString();
}
