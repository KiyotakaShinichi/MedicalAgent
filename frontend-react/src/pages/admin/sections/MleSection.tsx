import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { Badge } from "../../../components/ui/Badge";
import { statusVariant } from "../../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { MetricCard } from "../../../components/ui/MetricCard";
import { LoadingPane, ErrorPane, EmptyPane } from "../../../components/ui/Spinner";
import { useApi } from "../../../hooks/useApi";
import {
  runMleReadiness, getTrainingReport, getLockedHoldout,
  getExternalValidation, getModelComparison,
} from "../../../api/client";
import type { AdminAnalytics } from "../../../types/api";

interface Props { analytics: AdminAnalytics; onRefresh: () => void }

export function MleSection({ analytics, onRefresh }: Props) {
  const mle = analytics.mle_readiness;
  const [runningMle, setRunningMle] = useState(false);

  const { data: trainingReport, status: trStatus } = useApi(getTrainingReport, []);
  const { data: holdout, status: holdoutStatus } = useApi(getLockedHoldout, []);
  const { data: extVal, status: extValStatus } = useApi(getExternalValidation, []);
  const { data: modelComp, status: modelCompStatus } = useApi(getModelComparison, []);

  async function runMle() {
    setRunningMle(true);
    try { await runMleReadiness(); onRefresh(); } finally { setRunningMle(false); }
  }

  const tr = (trainingReport as { result?: Record<string, unknown> } | null)?.result;
  const ho = (holdout as { result?: Record<string, unknown> } | null)?.result;

  return (
    <div className="flex flex-col gap-4">
      {/* Status header */}
      <Card>
        <CardHeader>
          <SectionTitle>MLE Readiness</SectionTitle>
          <Button
            variant="secondary" size="sm"
            loading={runningMle}
            icon={<RefreshCw size={12} />}
            onClick={() => void runMle()}
          >
            Re-run gates
          </Button>
        </CardHeader>
        <div className="flex flex-wrap items-center gap-3 mb-3">
          <Badge variant={statusVariant(mle.status)}>{mle.status}</Badge>
          <span className="text-xs" style={{ color: "var(--text-dim)" }}>
            {mle.release_recommendation.replace(/_/g, " ")}
          </span>
          <span
            className="text-xs px-2 py-0.5 rounded border"
            style={{
              background: mle.hard_gate_failures === 0 ? "rgba(16,185,129,0.08)" : "rgba(244,63,94,0.08)",
              borderColor: mle.hard_gate_failures === 0 ? "rgba(16,185,129,0.25)" : "rgba(244,63,94,0.25)",
              color: mle.hard_gate_failures === 0 ? "var(--green)" : "var(--rose)",
            }}
          >
            {mle.hard_gate_failures === 0 ? "All hard gates passed" : `${mle.hard_gate_failures} gate failures`}
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Object.entries(mle.category_statuses).map(([cat, st]) => (
            <div key={cat} className="flex flex-col gap-1 p-2 rounded-md" style={{ background: "var(--surface2)" }}>
              <span className="text-xs" style={{ color: "var(--text-faint)" }}>{cat.replace(/_/g, " ")}</span>
              <Badge variant={statusVariant(st)}>{st}</Badge>
            </div>
          ))}
        </div>
      </Card>

      {/* Training report */}
      <Card>
        <CardHeader><SectionTitle>Training Report</SectionTitle></CardHeader>
        {trStatus === "loading" ? <LoadingPane /> :
         trStatus === "error" ? <ErrorPane message="Could not load training report" /> :
         !tr ? <EmptyPane label="No training report - run training first" /> : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              ["Test patients",      (tr as Record<string, unknown>).test_patients],
              ["Best classifier",    (tr as Record<string, unknown>).best_classifier],
              ["Best regressor",     (tr as Record<string, unknown>).best_regressor],
              ["AUROC",             (tr as Record<string, unknown>).auroc],
              ["Brier score",        (tr as Record<string, unknown>).brier_score],
              ["MAE",               (tr as Record<string, unknown>).mae],
              ["RMSE",              (tr as Record<string, unknown>).rmse],
            ].map(([label, val]) => (
              <MetricCard
                key={label as string}
                label={label as string}
                value={val != null ? String(val) : null}
              />
            ))}
          </div>
        )}
      </Card>

      {/* Locked holdout */}
      <Card>
        <CardHeader><SectionTitle>Locked Holdout Evaluation</SectionTitle></CardHeader>
        {holdoutStatus === "loading" ? <LoadingPane /> :
         !ho ? <EmptyPane label="No holdout evaluation - run holdout first" /> : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              ["AUROC",       (ho as Record<string, unknown>).auroc],
              ["Brier",       (ho as Record<string, unknown>).brier_score],
              ["Sensitivity", (ho as Record<string, unknown>).sensitivity],
              ["MAE",         (ho as Record<string, unknown>).mae],
            ].map(([label, val]) => (
              <MetricCard key={label as string} label={label as string} value={val != null ? String(val) : null} />
            ))}
          </div>
        )}
      </Card>

      {/* External validation */}
      <Card>
        <CardHeader><SectionTitle>External Validation Direction</SectionTitle></CardHeader>
        {extValStatus === "loading" ? <LoadingPane /> :
         !(extVal as { result?: unknown } | null)?.result
           ? <EmptyPane label="No external validation data" />
           : (
            <p className="text-xs" style={{ color: "var(--text-dim)" }}>
              External validation report loaded. See full JSON for dataset-level metrics.
            </p>
          )}
      </Card>

      {/* Model comparison */}
      <Card>
        <CardHeader><SectionTitle>Model Comparison</SectionTitle></CardHeader>
        {modelCompStatus === "loading" ? <LoadingPane /> :
         !(modelComp as { result?: unknown } | null)?.result
           ? <EmptyPane label="No model comparison available" />
           : (
            <p className="text-xs" style={{ color: "var(--text-dim)" }}>
              Model comparison loaded. Delta metrics available in JSON.
            </p>
          )}
      </Card>
    </div>
  );
}
