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
  getNoiseEval, getTemporalEval, getPredictionErrorTable,
  getPublicDataManifest,
} from "../../../api/client";
import type {
  AdminAnalytics,
  NoiseEvalResult,
  TemporalEvalResult,
  PredictionErrorTable,
  PublicDataManifest,
} from "../../../types/api";

interface Props { analytics: AdminAnalytics; onRefresh: () => void }

export function MleSection({ analytics, onRefresh }: Props) {
  const mle = analytics.mle_readiness;
  const [runningMle, setRunningMle] = useState(false);
  const [showGlossary, setShowGlossary] = useState(false);

  const { data: trainingReport, status: trStatus } = useApi(getTrainingReport, []);
  const { data: holdout, status: holdoutStatus } = useApi(getLockedHoldout, []);
  const { data: extVal, status: extValStatus } = useApi(getExternalValidation, []);
  const { data: modelComp, status: modelCompStatus } = useApi(getModelComparison, []);
  const { data: noiseEval, status: noiseStatus } = useApi(getNoiseEval, []);
  const { data: temporalEval, status: temporalStatus } = useApi(getTemporalEval, []);
  const { data: errorTable, status: errorStatus } = useApi(getPredictionErrorTable, []);
  const { data: dataManifest, status: dataManifestStatus } = useApi(getPublicDataManifest, []);

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

      {/* Public data feasibility */}
      <Card>
        <CardHeader>
          <SectionTitle>Public Data Feasibility</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(59,130,246,0.12)", color: "#93c5fd" }}>
            Source-calibrated synthetic data
          </span>
        </CardHeader>
        {dataManifestStatus === "loading" ? <LoadingPane /> :
         dataManifestStatus === "error" ? <ErrorPane message="Could not load public data manifest" /> :
         !dataManifest ? <EmptyPane label="No public data manifest available" /> : (
          <PublicDataManifestPanel data={dataManifest as PublicDataManifest} />
        )}
      </Card>

      {/* Noise robustness */}
      <Card>
        <CardHeader>
          <SectionTitle>Noise Robustness Evaluation</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(245,158,11,0.12)", color: "var(--amber)" }}>
            Synthetic perturbations
          </span>
        </CardHeader>
        {noiseStatus === "loading" ? <LoadingPane /> :
         noiseStatus === "error" ? <ErrorPane message="Could not load noise eval" /> :
         !noiseEval || (noiseEval as NoiseEvalResult).status === "unavailable" ? (
          <EmptyPane label="No noise eval — run POST /admin/noise-eval first" />
         ) : (
          <NoiseEvalPanel data={noiseEval as NoiseEvalResult} />
         )}
      </Card>

      {/* Temporal generalization */}
      <Card>
        <CardHeader>
          <SectionTitle>Temporal Generalization</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(59,130,246,0.12)", color: "#93c5fd" }}>
            Synthetic timeline splits
          </span>
        </CardHeader>
        {temporalStatus === "loading" ? <LoadingPane /> :
         temporalStatus === "error" ? <ErrorPane message="Could not load temporal eval" /> :
         !temporalEval || (temporalEval as TemporalEvalResult).status === "unavailable" ? (
          <EmptyPane label="No temporal eval — run POST /admin/temporal-eval first" />
         ) : (
          <TemporalEvalPanel data={temporalEval as TemporalEvalResult} />
         )}
      </Card>

      {/* Per-prediction ML error table */}
      <Card>
        <CardHeader>
          <SectionTitle>Per-Prediction Error Table</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(245,158,11,0.12)", color: "var(--amber)" }}>
            Synthetic holdout
          </span>
        </CardHeader>
        {errorStatus === "loading" ? <LoadingPane /> :
         errorStatus === "error" ? <ErrorPane message="Could not load prediction error table" /> :
         !errorTable || !(errorTable as PredictionErrorTable).rows?.length ? (
          <EmptyPane label="No predictions — run training pipeline first" />
         ) : (
          <PredictionErrorPanel data={errorTable as PredictionErrorTable} />
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

function PublicDataManifestPanel({ data }: { data: PublicDataManifest }) {
  const visibleNeeds = data.feature_feasibility.slice(0, 6);
  const sourceNames = new Map(data.sources.map((source) => [source.id, source.name]));

  return (
    <div className="flex flex-col gap-3">
      <div className="rounded-md border p-3" style={{ background: "var(--surface2)", borderColor: "var(--border)" }}>
        <p className="text-xs mb-2" style={{ color: "var(--text-dim)" }}>{data.central_data_reality}</p>
        <p className="text-xs" style={{ color: "var(--text-faint)" }}>{data.recommended_strategy}</p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Need", "Status", "Sources", "Project action"].map((h) => (
                <th key={h} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleNeeds.map((need) => (
              <tr key={need.need} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-4 font-medium" style={{ color: "var(--text)" }}>{need.need}</td>
                <td className="py-2 pr-4">
                  <Badge variant={
                    need.status === "covered_by_public_data" ? "green" :
                    need.status === "partially_covered" ? "amber" :
                    need.status === "future_extension" ? "blue" :
                    "red"
                  }>
                    {need.status.replace(/_/g, " ")}
                  </Badge>
                </td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>
                  {need.sources.length ? need.sources.map((id) => sourceNames.get(id) || id).join(", ") : "No direct public source"}
                </td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{need.project_action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid sm:grid-cols-2 gap-2">
        {data.sources.slice(0, 6).map((source) => (
          <div key={source.id} className="rounded-md border p-2" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
            <div className="flex items-center justify-between gap-2 mb-1">
              <p className="text-xs font-semibold" style={{ color: "var(--text)" }}>{source.name}</p>
              <span className="text-[10px]" style={{ color: "var(--text-faint)" }}>{source.provider}</span>
            </div>
            <p className="text-xs" style={{ color: "var(--text-dim)" }}>{source.use_in_project[0]}</p>
          </div>
        ))}
      </div>

      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>
        Manifest {data.manifest_hash}. {data.claim_boundary}
      </p>
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

function NoiseEvalPanel({ data }: { data: NoiseEvalResult }) {
  const base = data.clean_baseline;
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Baseline AUROC"    value={base.auroc != null ? base.auroc.toFixed(3) : null}    status="green" />
        <MetricCard label="Baseline Brier"    value={base.brier_score != null ? base.brier_score.toFixed(3) : null} status="green" />
        <MetricCard label="Baseline Sensitivity" value={base.sensitivity != null ? base.sensitivity.toFixed(3) : null} status="green" />
        <MetricCard label="Baseline PR-AUC"   value={base.pr_auc != null ? base.pr_auc.toFixed(3) : null}   status="green" />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Noise mode", "AUROC", "Δ AUROC", "Sensitivity", "Δ Sensitivity", "Status"].map(h => (
                <th key={h} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.noise_results.map((r) => (
              <tr key={r.mode} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-4 font-medium" style={{ color: "var(--text)" }}>{r.mode.replace(/_/g, " ")}</td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.auroc?.toFixed(3) ?? "—"}</td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: r.auroc_delta != null && r.auroc_delta < -0.05 ? "var(--rose)" : "var(--text-dim)" }}>
                  {r.auroc_delta != null ? (r.auroc_delta >= 0 ? "+" : "") + r.auroc_delta.toFixed(3) : "—"}
                </td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.sensitivity?.toFixed(3) ?? "—"}</td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: r.sensitivity_delta != null && r.sensitivity_delta < -0.05 ? "var(--rose)" : "var(--text-dim)" }}>
                  {r.sensitivity_delta != null ? (r.sensitivity_delta >= 0 ? "+" : "") + r.sensitivity_delta.toFixed(3) : "—"}
                </td>
                <td className="py-2">
                  <Badge variant={r.status === "robust" ? "green" : r.status === "degraded" ? "amber" : "red"}>{r.status}</Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.summary.worst_mode && (
        <p className="text-xs" style={{ color: "var(--text-faint)" }}>
          Worst mode: <strong style={{ color: "var(--text-dim)" }}>{data.summary.worst_mode.replace(/_/g, " ")}</strong>
          {data.summary.max_auroc_drop != null && ` · max AUROC drop ${data.summary.max_auroc_drop.toFixed(3)}`}
        </p>
      )}
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

function TemporalEvalPanel({ data }: { data: TemporalEvalResult }) {
  const splits = [
    { label: "Patient timeline split", metrics: data.temporal_split },
    { label: "Cycle accumulation split", metrics: data.cycle_split },
    { label: "Random baseline", metrics: data.random_split_baseline },
  ] as const;
  return (
    <div className="flex flex-col gap-3">
      <div className="grid sm:grid-cols-3 gap-3">
        {splits.map(({ label, metrics }) => (
          <div key={label} className="rounded-md border p-3" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
            <p className="text-xs font-semibold mb-2" style={{ color: "var(--text-dim)" }}>{label}</p>
            <div className="flex flex-col gap-1">
              <Row label="AUROC"       value={metrics.auroc?.toFixed(3)} />
              <Row label="Brier"       value={metrics.brier_score?.toFixed(3)} />
              <Row label="Sensitivity" value={metrics.sensitivity?.toFixed(3)} />
              <Row label="n train"     value={String(metrics.n_train)} />
              <Row label="n eval"      value={String(metrics.n_eval)} />
            </div>
          </div>
        ))}
      </div>
      {data.generalization_gap && (
        <div className="flex gap-4">
          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Temporal gap: <span style={{ color: "var(--text-dim)" }}>{data.generalization_gap.temporal_auroc_gap?.toFixed(3) ?? "—"}</span>
          </p>
          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Cycle gap: <span style={{ color: "var(--text-dim)" }}>{data.generalization_gap.cycle_auroc_gap?.toFixed(3) ?? "—"}</span>
          </p>
        </div>
      )}
      {data.interpretation && (
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>{data.interpretation}</p>
      )}
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string | undefined }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-xs" style={{ color: "var(--text-faint)" }}>{label}</span>
      <span className="text-xs tabular-nums font-medium" style={{ color: "var(--text-dim)" }}>{value ?? "—"}</span>
    </div>
  );
}

const CONFUSION_COLOR: Record<string, string> = {
  TP: "var(--green)", FP: "var(--amber)", TN: "var(--text-dim)", FN: "var(--rose)"
};

function PredictionErrorPanel({ data }: { data: PredictionErrorTable }) {
  const [showAll, setShowAll] = useState(false);
  const rows = showAll ? data.rows : data.rows.slice(0, 20);
  const cm = data.confusion_summary;

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Sensitivity" value={data.sensitivity != null ? data.sensitivity.toFixed(3) : null}
          status={data.sensitivity != null && data.sensitivity >= 0.75 ? "green" : "amber"} />
        <MetricCard label="Specificity" value={data.specificity != null ? data.specificity.toFixed(3) : null} />
        <MetricCard label="MAE"         value={data.mae != null ? data.mae.toFixed(4) : null} />
        <MetricCard label="Threshold"   value={String(data.threshold)} />
      </div>
      <div className="flex gap-4">
        {(["TP", "FP", "TN", "FN"] as const).map(k => (
          <div key={k} className="flex flex-col items-center gap-0.5">
            <span className="text-lg font-bold tabular-nums" style={{ color: CONFUSION_COLOR[k] }}>{cm[k]}</span>
            <span className="text-xs" style={{ color: "var(--text-faint)" }}>{k}</span>
          </div>
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["ID", "Actual", "Prob", "Class", "Error", "Type"].map(h => (
                <th key={h} className="text-left py-2 pr-3 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.patient_id} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-1.5 pr-3" style={{ color: "var(--text-dim)" }}>{r.patient_id}</td>
                <td className="py-1.5 pr-3 tabular-nums" style={{ color: "var(--text)" }}>{r.actual_label}</td>
                <td className="py-1.5 pr-3 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.predicted_probability.toFixed(3)}</td>
                <td className="py-1.5 pr-3 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.predicted_class}</td>
                <td className="py-1.5 pr-3 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.absolute_error.toFixed(4)}</td>
                <td className="py-1.5">
                  <span className="font-bold" style={{ color: CONFUSION_COLOR[r.confusion_type] }}>{r.confusion_type}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.rows.length > 20 && (
        <Button variant="ghost" size="sm" onClick={() => setShowAll(v => !v)}>
          {showAll ? `Show fewer` : `Show all ${data.rows.length} rows`}
        </Button>
      )}
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}
