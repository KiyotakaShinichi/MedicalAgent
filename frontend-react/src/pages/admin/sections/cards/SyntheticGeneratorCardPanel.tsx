import { RefreshCw, Info } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { statusVariant } from "../../../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { Button } from "../../../../components/ui/Button";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { LoadingPane, EmptyPane } from "../../../../components/ui/Spinner";
import type { SyntheticGeneratorCard } from "../../../../types/api";

/**
 * Phase 6 — Synthetic generator card.
 *
 * Documents the generator assumptions, known shortcuts, and unsupported
 * claims. Keeping it isolated makes the synthetic-only claim boundary
 * easy to review without spelunking through the full admin section.
 */
export function SyntheticGeneratorCardPanel({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: SyntheticGeneratorCard | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const cohort = report?.cohort ?? {};
  const dist = report?.feature_distribution_summary ?? {};

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Info size={14} style={{ color: "var(--blue, #1e3a8a)" }} aria-hidden="true" />
          <SectionTitle>Synthetic generator card</SectionTitle>
          <Badge variant={statusVariant(status === "passed" ? "strong" : status === "missing" ? "stale" : "needs_attention")}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Rebuilding..." : "Rebuild"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Documents what the synthetic dataset is, the causal assumptions baked
        into the generator, known shortcuts, and what cannot be claimed from
        these numbers. Reviewer-facing provenance.
      </p>

      {loading ? (
        <LoadingPane label="Loading generator card..." />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Generator card has not been built yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-4 gap-3 mb-3">
            <MetricCard label="Patients" value={cohort.patients_created ?? 0} status="muted" />
            <MetricCard label="Rows" value={dist.row_count ?? 0} status="muted" />
            <MetricCard
              label="Positive label rate"
              value={dist.positive_label_rate != null ? `${(dist.positive_label_rate * 100).toFixed(1)}%` : "-"}
              status="muted"
            />
            <MetricCard
              label="Card -> dataset"
              value={report.card_version_matches_dataset ? "in sync" : "drifted"}
              status={report.card_version_matches_dataset ? "green" : "amber"}
            />
          </div>

          {report.causal_assumptions && report.causal_assumptions.length > 0 && (
            <ProvenanceNarrative title="Causal assumptions" items={report.causal_assumptions} tone="info" />
          )}
          {report.known_shortcuts && report.known_shortcuts.length > 0 && (
            <ProvenanceNarrative title="Known shortcuts the model could exploit" items={report.known_shortcuts} tone="amber" />
          )}
          {report.unsupported_claims && report.unsupported_claims.length > 0 && (
            <ProvenanceNarrative title="What this dataset CANNOT support claiming" items={report.unsupported_claims} tone="rose" />
          )}

          <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
            Schema: <code>{report.dataset_schema_version ?? "unknown"}</code>{" · "}
            Card: <code>{report.generator_card_version ?? "unknown"}</code>{" · "}
            Rows fingerprint: <code>{cohort.rows_fingerprint ?? "-"}</code>
          </p>
        </>
      )}
    </Card>
  );
}

function ProvenanceNarrative({ title, items, tone }: { title: string; items: string[]; tone: "info" | "amber" | "rose" }) {
  const palette = tone === "amber"
    ? { fg: "#92400e", bg: "rgba(245,158,11,0.06)", border: "rgba(245,158,11,0.25)" }
    : tone === "rose"
      ? { fg: "#b91c1c", bg: "rgba(244,63,94,0.05)", border: "rgba(244,63,94,0.25)" }
      : { fg: "var(--text)", bg: "var(--surface2)", border: "var(--border)" };
  return (
    <div
      className="rounded-md border p-3 mb-2"
      style={{ background: palette.bg, borderColor: palette.border }}
    >
      <p className="text-[0.72rem] uppercase font-semibold mb-1.5" style={{ color: palette.fg }}>
        {title}
      </p>
      <ul className="flex flex-col gap-1 pl-4" style={{ listStyle: "disc", color: "var(--text)" }}>
        {items.map((item, i) => (
          <li key={i} className="text-xs leading-relaxed">{item}</li>
        ))}
      </ul>
    </div>
  );
}
