import { ShieldCheck, Database, Star, Cpu, Activity } from "lucide-react";
import { MetricCard } from "../../../components/ui/MetricCard";
import { Badge } from "../../../components/ui/Badge";
import { statusVariant } from "../../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { useApi } from "../../../hooks/useApi";
import { getNormalizedBenchmarkArtifact } from "../../../api/client";
import type { AdminAnalytics } from "../../../types/api";

interface Props { analytics: AdminAnalytics }

export function OverviewSection({ analytics }: Props) {
  const { rag_evaluation: rag, guardrails, mle_readiness: mle, agent_feedback: fb } = analytics;
  const { data: credibility } = useApi(
    () => getNormalizedBenchmarkArtifact("credibility_gap_registry"),
    [],
  );
  const { data: finetune } = useApi(
    () => getNormalizedBenchmarkArtifact("finetune_hardening_assurance"),
    [],
  );
  const { data: evidenceMaturity } = useApi(
    () => getNormalizedBenchmarkArtifact("evidence_maturity_matrix"),
    [],
  );
  const { data: xaiReliability } = useApi(
    () => getNormalizedBenchmarkArtifact("xai_reliability_gate"),
    [],
  );

  return (
    <div className="flex flex-col gap-6">
      {/* Top metric row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          label="RAG grounding"
          value={rag.grounding_score != null ? `${(rag.grounding_score * 100).toFixed(1)}%` : null}
          icon={Database}
          status={rag.grounding_score != null && rag.grounding_score >= 0.8 ? "green" : "amber"}
        />
        <MetricCard
          label="Attack block rate"
          value={guardrails.attack_block_rate != null ? `${(guardrails.attack_block_rate * 100).toFixed(0)}%` : null}
          icon={ShieldCheck}
          status={guardrails.attack_block_rate === 1 ? "green" : guardrails.attack_block_rate != null && guardrails.attack_block_rate >= 0.9 ? "amber" : "red"}
        />
        <MetricCard
          label="Agent feedback"
          value={fb.average_rating != null ? fb.average_rating.toFixed(1) : null}
          unit="/5"
          icon={Star}
          status={fb.average_rating != null && fb.average_rating >= 4 ? "green" : "amber"}
          sub={`${fb.count} responses`}
        />
        <MetricCard
          label="MLE status"
          value={mle.status}
          icon={Cpu}
          status={statusVariant(mle.status) === "green" ? "green" : statusVariant(mle.status) === "amber" ? "amber" : "red"}
        />
      </div>

      {/* MLE category grid */}
      <Card>
        <CardHeader><SectionTitle>MLE Release Gates</SectionTitle></CardHeader>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Object.entries(mle.category_statuses).map(([cat, status]) => (
            <div key={cat} className="flex flex-col gap-1 p-2 rounded-md" style={{ background: "var(--surface2)" }}>
              <span className="text-xs" style={{ color: "var(--text-faint)" }}>
                {cat.replace(/_/g, " ")}
              </span>
              <Badge variant={statusVariant(status)}>{status}</Badge>
            </div>
          ))}
        </div>
        <div className="mt-3 pt-3 border-t flex items-center gap-3" style={{ borderColor: "var(--border)" }}>
          <span className="text-xs" style={{ color: "var(--text-dim)" }}>Release recommendation:</span>
          <Badge variant={statusVariant(mle.release_recommendation)}>
            {mle.release_recommendation.replace(/_/g, " ")}
          </Badge>
        </div>
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle>Evidence Credibility</SectionTitle>
          <Badge variant="amber">internal engineering evidence</Badge>
        </CardHeader>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard
            label="Open evidence gaps"
            value={metricNumber(credibility?.metrics.open_or_external_count)}
            status="amber"
          />
          <MetricCard
            label="Not self-certifiable"
            value={metricNumber(credibility?.metrics.cannot_be_self_certified_count)}
            status="amber"
          />
          <MetricCard
            label="Fine-tune decision"
            value={metricText(finetune?.metrics.promotion_decision) ?? "HOLD"}
            status="amber"
          />
          <MetricCard
            label="Fine-tune blockers"
            value={metricNumber(finetune?.metrics.blocking_gap_count)}
            status="amber"
          />
          <MetricCard
            label="Architecture hotspots"
            value={metricNumber(evidenceMaturity?.metrics.oversized_file_count)}
            status="amber"
          />
          <MetricCard
            label="Critical large files"
            value={metricNumber(evidenceMaturity?.metrics.critical_file_count)}
            status="amber"
          />
          <MetricCard
            label="XAI display policy"
            value={xaiModeLabel(metricText(xaiReliability?.metrics.display_mode))}
            status="amber"
          />
          <MetricCard
            label="Rank claims allowed"
            value={metricBoolean(xaiReliability?.metrics.ranked_feature_order_allowed)}
            status="amber"
          />
        </div>
        <p className="text-xs mt-3" style={{ color: "var(--text-dim)" }}>
          These are evidence-completion fields, not clinical-readiness or safety scores.
        </p>
      </Card>

      {/* Quick stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Evaluations"   value={rag.evaluations}   icon={Activity} />
        <MetricCard label="Cache hit rate" value={rag.cache_hit_rate != null ? `${(rag.cache_hit_rate * 100).toFixed(0)}%` : null} />
        <MetricCard label="Input blocks"  value={guardrails.input_blocks} />
        <MetricCard label="Output blocks" value={guardrails.output_blocks} />
      </div>
    </div>
  );
}

function metricNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function metricText(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function metricBoolean(value: unknown): string | null {
  return typeof value === "boolean" ? (value ? "yes" : "no") : null;
}

function xaiModeLabel(value: string | null): string | null {
  if (value === "ranked_factors_with_noncausal_boundary") return "ranked";
  if (value === "grouped_factors_without_rank_claim") return "grouped only";
  if (value === "suppress_feature_factors") return "hidden";
  return value?.replace(/_/g, " ") ?? null;
}
