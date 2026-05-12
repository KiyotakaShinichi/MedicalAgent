import { useState } from "react";
import { RefreshCw, AlertTriangle, Info } from "lucide-react";
import { Badge } from "../../../components/ui/Badge";
import { statusVariant } from "../../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { MetricCard } from "../../../components/ui/MetricCard";
import { MetricGlossary, ALL_METRIC_SPECS } from "../../../components/ui/MetricInterpretation";
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
  const [showGlossary, setShowGlossary] = useState(false);

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

  const trValues: Record<string, number | null> = tr ? {
    "AUROC": parseFloat(String(tr.auroc)) || null,
    "Brier Score": parseFloat(String(tr.brier_score)) || null,
    "ECE": parseFloat(String(tr.ece)) || null,
    "Sensitivity (Recall)": parseFloat(String(tr.sensitivity)) || null,
    "MAE (Regression)": parseFloat(String(tr.mae)) || null,
  } : {};

  return (
    <div className="flex flex-col gap-4">
      {/* Synthetic data disclaimer */}
      <div
        className="flex items-start gap-2 px-3 py-2.5 rounded-lg border text-xs"
        style={{ background: "rgba(245,158,11,0.07)", borderColor: "rgba(245,158,11,0.25)", color: "var(--amber)" }}
      >
        <AlertTriangle size={13} style={{ flexShrink: 0, marginTop: 1 }} />
        <span>
          All metrics below are computed on <strong>synthetic data</strong> unless explicitly labelled "locked holdout" or "external validation".
          Synthetic AUROC is expected to be high and does not reflect clinical validity.
          The locked holdout uses a frozen synthetic split; external validation uses BreastDCEDL/I-SPY1 tabular features.
        </span>
      </div>

      {/* Cost-sensitive eval rationale */}
      <Card>
        <CardHeader>
          <SectionTitle>Threshold &amp; Cost-Sensitive Evaluation</SectionTitle>
          <Info size={14} style={{ color: "var(--text-faint)" }} />
        </CardHeader>
        <div className="grid sm:grid-cols-3 gap-3 mb-3">
          <CostCard
            label="False Negative Cost"
            level="HIGH"
            color="var(--rose)"
            description="Missed positive case (failed response) delays clinician intervention. Prioritise recall."
          />
          <CostCard
            label="False Positive Cost"
            level="MODERATE"
            color="var(--amber)"
            description="Unnecessary flag increases review burden. Acceptable trade-off for lower FNR."
          />
          <CostCard
            label="Operating Threshold"
            level="≤ 0.40"
            color="var(--blue)"
            description="Decision threshold set below 0.50 to bias toward sensitivity. Reviewed per-model at training time."
          />
        </div>
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>
          This system uses a cost-sensitive approach: the classification threshold is chosen to minimise FNR at acceptable FPR,
          reflecting the assumption that missing a treatment non-response is more harmful than over-flagging for clinician review.
          Weighted cost = FN_weight × FN + FP_weight × FP where FN_weight = 3, FP_weight = 1 (engineering heuristic, not clinical guidance).
        </p>
      </Card>

      {/* Gate status */}
      <Card>
        <CardHeader>
          <SectionTitle>MLE Readiness Gates</SectionTitle>
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
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
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
        <CardHeader>
          <SectionTitle>Synthetic Training Report</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(245,158,11,0.12)", color: "var(--amber)" }}>
            Synthetic data
          </span>
        </CardHeader>
        {trStatus === "loading" ? <LoadingPane /> :
         trStatus === "error" ? <ErrorPane message="Could not load training report" /> :
         !tr ? <EmptyPane label="No training report — run training first" /> : (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
              {[
                ["Test patients",   tr.test_patients,   null,  "muted"],
                ["Best classifier", tr.best_classifier, null,  "muted"],
                ["Best regressor",  tr.best_regressor,  null,  "muted"],
                ["AUROC",          tr.auroc,            0.85,  "green"],
                ["Brier score",     tr.brier_score,     0.10,  "green"],
                ["MAE",            tr.mae,              0.10,  "green"],
                ["RMSE",           tr.rmse,             0.15,  "green"],
              ].map(([label, val, threshold]) => {
                const num = typeof val === "number" ? val : parseFloat(String(val));
                const status: "green" | "amber" | "red" | "muted" =
                  threshold == null || isNaN(num) ? "muted" :
                  label === "Brier score" || label === "MAE" || label === "RMSE"
                    ? (num <= (threshold as number) ? "green" : num <= (threshold as number) * 2 ? "amber" : "red")
                    : (num >= (threshold as number) ? "green" : num >= (threshold as number) * 0.85 ? "amber" : "red");
                return (
                  <MetricCard
                    key={label as string}
                    label={label as string}
                    value={val != null ? String(val) : null}
                    status={status}
                  />
                );
              })}
            </div>
            <p className="text-xs" style={{ color: "var(--text-faint)" }}>
              Metrics labelled with interpretation bands below. AUROC ≥ 0.85 = strong on synthetic; Brier &lt; 0.10 = well-calibrated; MAE &lt; 0.10 = good regression fit.
            </p>
          </>
        )}
      </Card>

      {/* Locked holdout */}
      <Card>
        <CardHeader>
          <SectionTitle>Locked Holdout Evaluation</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(59,130,246,0.12)", color: "#93c5fd" }}>
            Frozen synthetic split
          </span>
        </CardHeader>
        {holdoutStatus === "loading" ? <LoadingPane /> :
         !ho ? <EmptyPane label="No holdout evaluation — run holdout first" /> : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              ["AUROC",       ho.auroc,        0.80, "Higher = better discrimination"],
              ["Brier",       ho.brier_score,  0.12, "Lower = better calibration"],
              ["Sensitivity", ho.sensitivity,  0.75, "Higher = fewer missed positives"],
              ["MAE",         ho.mae,          0.12, "Lower = better regression fit"],
            ].map(([label, val, threshold, tip]) => {
              const num = typeof val === "number" ? val : parseFloat(String(val));
              const isLowerBetter = label === "Brier" || label === "MAE";
              const status: "green" | "amber" | "red" | "muted" = isNaN(num) ? "muted" :
                isLowerBetter
                  ? (num <= (threshold as number) ? "green" : num <= (threshold as number) * 1.6 ? "amber" : "red")
                  : (num >= (threshold as number) ? "green" : num >= (threshold as number) * 0.85 ? "amber" : "red");
              return (
                <MetricCard
                  key={label as string}
                  label={label as string}
                  value={val != null ? String(val) : null}
                  status={status}
                  sub={tip as string}
                />
              );
            })}
          </div>
        )}
      </Card>

      {/* External validation */}
      <Card>
        <CardHeader>
          <SectionTitle>External Validation Direction</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(139,92,246,0.12)", color: "#c4b5fd" }}>
            BreastDCEDL / I-SPY1
          </span>
        </CardHeader>
        {extValStatus === "loading" ? <LoadingPane /> :
         !(extVal as { result?: unknown } | null)?.result
           ? <EmptyPane label="No external validation data — run external validation first" />
           : (
            <div className="flex flex-col gap-2">
              <p className="text-xs" style={{ color: "var(--text-dim)" }}>
                External validation uses BreastDCEDL and I-SPY1 tabular MRI-derived features.
                These are real datasets (non-synthetic) used for directional validation only —
                not a clinical performance claim.
              </p>
              <p className="text-xs px-2 py-1.5 rounded-md border" style={{
                background: "rgba(139,92,246,0.07)", borderColor: "rgba(139,92,246,0.25)", color: "#c4b5fd"
              }}>
                ✓ External validation report loaded. See <code>Data/external_validation/</code> for per-dataset metrics.
              </p>
            </div>
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
              Model comparison loaded. Δ AUROC, Δ Brier, Δ ECE, Δ FNR deltas available in <code>Data/model_comparison/</code>.
            </p>
          )}
      </Card>

      {/* Metric glossary toggle */}
      <Card>
        <CardHeader>
          <SectionTitle>Metric Interpretation Guide</SectionTitle>
          <Button
            variant="ghost" size="sm"
            onClick={() => setShowGlossary((v) => !v)}
          >
            {showGlossary ? "Hide" : "Show"} guide
          </Button>
        </CardHeader>
        {showGlossary ? (
          <div className="flex flex-col gap-2">
            <p className="text-xs mb-2" style={{ color: "var(--text-dim)" }}>
              Each metric below shows its definition, why it matters in this context, and ideal / warning / bad interpretation bands.
              Bands reflect engineering heuristics for a cancer monitoring PoC — not clinical validation thresholds.
            </p>
            <MetricGlossary specs={ALL_METRIC_SPECS} values={trValues} />
          </div>
        ) : (
          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Click "Show guide" to see interpretation bands for AUROC, PR-AUC, Brier, ECE, Sensitivity, Specificity, FNR, and MAE.
          </p>
        )}
      </Card>
    </div>
  );
}

function CostCard({ label, level, color, description }: {
  label: string; level: string; color: string; description: string;
}) {
  return (
    <div className="rounded-md border p-3" style={{
      background: `${color}0d`, borderColor: `${color}30`,
    }}>
      <p className="text-xs font-semibold mb-0.5" style={{ color: "var(--text-faint)" }}>{label}</p>
      <p className="text-lg font-bold mb-1" style={{ color }}>{level}</p>
      <p className="text-xs" style={{ color: "var(--text-dim)" }}>{description}</p>
    </div>
  );
}
